import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import multer from 'multer';
import pdfParse from 'pdf-parse';
import dotenv from 'dotenv';
import Groq from 'groq-sdk';
import { GoogleGenAI } from '@google/genai';
import fs from 'fs';
import path from 'path';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import { rateLimit } from 'express-rate-limit';
import { connectDatabase, User, Case, SavedCase, ChatSession as ChatSessionModel, PasswordResetOtp } from './database';

// Initialize environment variables
dotenv.config();

const app = express();
const port = process.env.PORT || 5000;

// Trust the first proxy hop — required for Render/Heroku/Vercel deployments
// so that express-rate-limit can correctly read the real client IP from X-Forwarded-For
app.set('trust proxy', 1);

// Enable CORS and JSON parsing
app.use(cors());
app.use(express.json());

// Define Rate Limiters to secure API endpoints against brute force and API abuse
const generalLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
  legacyHeaders: false, // Disable the `X-RateLimit-*` headers
  message: {
    error: 'Too many requests from this IP, please try again after 15 minutes.'
  }
});

const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 15, // Limit each IP to 15 login/register attempts per windowMs
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: 'Too many login or registration attempts. Please try again after 15 minutes.'
  }
});

// Global minute-based AI rate limiter to protect the upstream 15 RPM limit (Gemini Studio / Groq key bottleneck)
const aiMinutlyLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 15, // Max 15 requests per minute globally
  keyGenerator: (req) => 'global_ai_minutly', // Applies globally across all users/IPs to prevent key suspension
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: 'AI service rate limit reached (15 RPM). Please wait a moment before trying again.'
  }
});

// Global daily AI rate limiter to protect the upstream 500 RPD limit (Gemini Studio bottleneck)
const aiDailyLimiter = rateLimit({
  windowMs: 24 * 60 * 60 * 1000, // 24 hours
  max: 500, // Max 500 requests per day globally
  keyGenerator: (req) => 'global_ai_daily', // Applies globally across all users/IPs
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: 'Daily AI query quota reached (500 RPD). Please try again tomorrow.'
  }
});

// Apply the general rate limiter to all API endpoints
app.use('/api/', generalLimiter);

// Set up Multer for handling file uploads (stored in memory)
// Enforces a strict 10MB limit to prevent memory exhaustion from large scans
const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB max
});

// Initialize Groq SDK
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY || '',
});

// Initialize Gemini SDK helper (reads live process.env.GEMINI_API_KEY on call)
const getGeminiClient = () => {
  const apiKey = process.env.GEMINI_API_KEY || '';
  return apiKey.trim() ? new GoogleGenAI({ apiKey: apiKey.trim() }) : null;
};

// Helper: safe JSON extraction from model response
function extractJSON(text: string): any {
  try {
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      // Clean up trailing commas in arrays/objects to prevent parsing errors
      const cleaned = jsonMatch[0].replace(/,\s*([\]}])/g, '$1');
      return JSON.parse(cleaned);
    }
    const cleanedText = text.replace(/,\s*([\]}])/g, '$1');
    return JSON.parse(cleanedText);
  } catch (err) {
    console.error('Failed to parse JSON directly. Raw response:', text);
    throw new Error('AI response did not contain valid structured JSON data.');
  }
}

// ─── DATABASE INITIALIZATION ──────────────────────────────────────────────────
connectDatabase();

// In-memory Reference Cache to avoid redundant Groq LLM queries for the same case/report
// Prefixed with userId to ensure multi-tenant cache isolation
const referenceCache = new Map<string, any>();

// ─── AUTHENTICATION MIDDLEWARE & INTERFACES ───────────────────────────────────

export interface AuthRequest extends Request {
  user?: { id: string; email: string };
}

export function authenticateToken(req: AuthRequest, res: Response, next: NextFunction) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required.', code: 'TOKEN_REQUIRED' });
  }

  jwt.verify(token, process.env.JWT_SECRET || 'scanwise-jwt-secret-key-2026', (err: any, decoded: any) => {
    if (err) {
      return res.status(401).json({
        error: 'Access token is invalid.',
        code: 'TOKEN_INVALID'
      });
    }
    req.user = decoded;
    next();
  });
}

// ─── AUTHENTICATION ENDPOINTS ──────────────────────────────────────────────────

/**
 * POST /api/auth/register
 * Registers a new clinician account with a hashed password. Returns a JWT.
 */
app.post('/api/auth/register', authLimiter, async (req: Request, res: Response): Promise<any> => {
  try {
    const { name, email, password, specialty, institution } = req.body;

    if (!name || !email || !password || !specialty || !institution) {
      return res.status(400).json({ error: 'All registration fields are required.' });
    }

    const emailLower = email.trim().toLowerCase();

    // 1. Backend email format validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(emailLower)) {
      return res.status(400).json({ error: 'Please enter a valid clinical email address.' });
    }

    // 2. Backend password length validation
    if (password.length < 6) {
      return res.status(400).json({ error: 'Password must be at least 6 characters long.' });
    }

    const existingUser = await User.findOne({ email: emailLower });
    if (existingUser) {
      return res.status(400).json({ error: 'A clinician with this email already exists.' });
    }

    // Hash password with bcrypt
    const hashedPassword = await bcrypt.hash(password, 10);

    const newUser = new User({
      name: name.trim(),
      email: emailLower,
      password: hashedPassword,
      specialty: specialty.trim(),
      institution: institution.trim(),
    });

    await newUser.save();

    // Sign JWT (valid indefinitely)
    const token = jwt.sign(
      { id: newUser._id.toString(), email: newUser.email },
      process.env.JWT_SECRET || 'scanwise-jwt-secret-key-2026'
    );

    console.log(`[Auth] Registered new surgeon: ${newUser.email} (${newUser._id})`);

    res.json({
      success: true,
      token,
      userProfile: {
        name: newUser.name,
        specialty: newUser.specialty,
        institution: newUser.institution,
      },
    });
  } catch (err: any) {
    console.error('Error in /api/auth/register:', err);
    res.status(500).json({ error: 'Internal server error during registration.' });
  }
});

/**
 * POST /api/auth/login
 * Authenticates clinician credentials. Returns a JWT.
 */
app.post('/api/auth/login', authLimiter, async (req: Request, res: Response): Promise<any> => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required.' });
    }

    const emailLower = email.trim().toLowerCase();
    const user = await User.findOne({ email: emailLower });
    if (!user) {
      return res.status(401).json({ error: 'Invalid email or password.' });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(401).json({ error: 'Invalid email or password.' });
    }

    // Sign JWT (valid indefinitely)
    const token = jwt.sign(
      { id: user._id.toString(), email: user.email },
      process.env.JWT_SECRET || 'scanwise-jwt-secret-key-2026'
    );

    console.log(`[Auth] Surgeon logged in successfully: ${user.email}`);

    res.json({
      success: true,
      token,
      userProfile: {
        name: user.name,
        specialty: user.specialty,
        institution: user.institution,
      },
    });
  } catch (err: any) {
    console.error('Error in /api/auth/login:', err);
    res.status(500).json({ error: 'Internal server error during login.' });
  }
});

// ─── GMAIL REST API HELPERS ──────────────────────────────────────────────────
// Pure HTTPS (port 443) — no SMTP — works on Railway & Render

// Warn on startup if Gmail credentials are missing — check Railway environment variables
const GMAIL_VARS_OK =
  !!process.env.GMAIL_CLIENT_ID &&
  !!process.env.GMAIL_CLIENT_SECRET &&
  !!process.env.GMAIL_REFRESH_TOKEN &&
  !!process.env.EMAIL_USER;
if (!GMAIL_VARS_OK) {
  console.warn(
    '[Gmail] ⚠️  One or more Gmail env vars are missing (GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN, EMAIL_USER). ' +
    'Forgot-password emails will fail. Set these in your Railway / Render environment variables.'
  );
}

async function getGmailAccessToken(): Promise<string> {
  if (!GMAIL_VARS_OK) {
    throw new Error('Gmail credentials are not configured on this server. Contact the administrator.');
  }
  const res = await fetch('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: process.env.GMAIL_CLIENT_ID || '',
      client_secret: process.env.GMAIL_CLIENT_SECRET || '',
      refresh_token: process.env.GMAIL_REFRESH_TOKEN || '',
      grant_type: 'refresh_token',
    }),
  });
  const data: any = await res.json();
  if (!data.access_token) {
    // Log the full OAuth error server-side but surface a clean message to the client
    console.error('[Gmail] OAuth token fetch failed:', JSON.stringify(data));
    const hint = data.error === 'invalid_grant'
      ? 'The Gmail refresh token has expired. Please re-authorise the Gmail OAuth app and update GMAIL_REFRESH_TOKEN.'
      : `Gmail OAuth error: ${data.error || 'unknown'} — ${data.error_description || ''}`;
    throw new Error(hint);
  }
  return data.access_token;
}

async function sendGmailEmail(to: string, subject: string, htmlBody: string): Promise<void> {
  const accessToken = await getGmailAccessToken();
  const rawMessage = [
    `From: OcnoDetect <${process.env.EMAIL_USER}>`,
    `To: ${to}`,
    `Subject: ${subject}`,
    `MIME-Version: 1.0`,
    `Content-Type: text/html; charset=utf-8`,
    ``,
    htmlBody,
  ].join('\r\n');
  const encoded = Buffer.from(rawMessage).toString('base64url');
  const res = await fetch('https://gmail.googleapis.com/gmail/v1/users/me/messages/send', {
    method: 'POST',
    headers: { Authorization: `Bearer ${accessToken}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ raw: encoded }),
  });
  if (!res.ok) {
    const err: any = await res.json().catch(() => ({}));
    throw new Error(err?.error?.message || `Gmail API send failed (${res.status})`);
  }
}

// ─── FORGOT PASSWORD ENDPOINTS ────────────────────────────────────────────────

/**
 * POST /api/auth/forgot-password
 * Generates a 6-digit OTP, saves it to DB (expires 10 min), emails it to the clinician.
 */
app.post('/api/auth/forgot-password', authLimiter, async (req: Request, res: Response): Promise<any> => {
  try {
    const { email } = req.body;
    if (!email) {
      return res.status(400).json({ error: 'Email is required.' });
    }

    const emailLower = email.trim().toLowerCase();
    const user = await User.findOne({ email: emailLower });

    // Always return success to avoid user enumeration — don't reveal if email exists
    if (!user) {
      return res.json({ success: true, message: 'If this email is registered, an OTP has been sent.' });
    }

    // Generate a 6-digit OTP
    const otp = Math.floor(100000 + Math.random() * 900000).toString();
    const expiresAt = new Date(Date.now() + 10 * 60 * 1000); // 10 minutes

    // Delete any existing OTP for this email, then save the new one
    await PasswordResetOtp.deleteMany({ email: emailLower });
    await PasswordResetOtp.create({ email: emailLower, otp, expiresAt });

    // Send OTP via Gmail REST API (pure HTTPS — not SMTP, works on Railway)
    try {
      await sendGmailEmail(
        emailLower,
        'Your OcnoDetect Password Reset OTP',
        `<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 480px; margin: auto; border: 1px solid #e2e8f0; border-radius: 16px; overflow: hidden;">
          <div style="background: #0ea5e9; padding: 28px 32px; text-align: center;">
            <span style="font-size: 22px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;">Ocno<span style="color: #bae6fd;">Detect</span></span>
            <p style="color: #e0f2fe; font-size: 13px; margin: 6px 0 0 0;">Clinical Intelligence Platform</p>
          </div>
          <div style="padding: 32px;">
            <h2 style="color: #0f172a; font-size: 20px; font-weight: 700; margin: 0 0 8px 0;">Password Reset Request</h2>
            <p style="color: #64748b; font-size: 14px; line-height: 22px; margin: 0 0 28px 0;">We received a request to reset your OcnoDetect account password. Use the one-time code below. It expires in <strong style="color: #0f172a;">10 minutes</strong>.</p>
            <div style="background: #f0f9ff; border: 2px solid #0ea5e9; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 28px;">
              <p style="color: #0ea5e9; font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin: 0 0 12px 0;">Your OTP Code</p>
              <div style="font-size: 40px; font-weight: 800; color: #0f172a; letter-spacing: 12px;">${otp}</div>
            </div>
            <div style="background: #fefce8; border-left: 3px solid #eab308; border-radius: 6px; padding: 12px 16px; margin-bottom: 24px;">
              <p style="color: #713f12; font-size: 13px; margin: 0;">⚠️ If you did not request this, ignore this email. Your password will remain unchanged.</p>
            </div>
            <p style="color: #94a3b8; font-size: 12px; margin: 0;">This code is valid for a single use only and will expire after 10 minutes.</p>
          </div>
          <div style="background: #f8fafc; border-top: 1px solid #e2e8f0; padding: 20px 32px; text-align: center;">
            <p style="color: #94a3b8; font-size: 12px; margin: 0;">© 2026 OcnoDetect · Clinical Intelligence Platform</p>
            <p style="color: #cbd5e1; font-size: 11px; margin: 4px 0 0 0;">Do not reply to this email · This is an automated message</p>
          </div>
        </div>`
      );
      console.log(`[Auth] Password reset OTP sent to: ${emailLower}`);
    } catch (emailErr: any) {
      console.error('[Auth] Failed to send OTP email:', emailErr.message);
      // Clean up the OTP so it can't be used without delivery
      await PasswordResetOtp.deleteMany({ email: emailLower }).catch(() => { });
      return res.status(503).json({
        error: 'Unable to send OTP email at this time. Please try again later or contact support.',
      });
    }

    res.json({ success: true, message: 'If this email is registered, an OTP has been sent.' });
  } catch (err: any) {
    console.error('Error in /api/auth/forgot-password:', err);
    res.status(500).json({ error: 'An unexpected error occurred. Please try again.' });
  }
});

/**
 * POST /api/auth/verify-otp
 * Verifies the 6-digit OTP is correct and not expired.
 */
app.post('/api/auth/verify-otp', authLimiter, async (req: Request, res: Response): Promise<any> => {
  try {
    const { email, otp } = req.body;
    if (!email || !otp) {
      return res.status(400).json({ error: 'Email and OTP are required.' });
    }

    const emailLower = email.trim().toLowerCase();
    const record = await PasswordResetOtp.findOne({ email: emailLower });

    if (!record) {
      return res.status(400).json({ error: 'OTP not found or already used. Please request a new one.' });
    }

    if (new Date() > record.expiresAt) {
      await PasswordResetOtp.deleteMany({ email: emailLower });
      return res.status(400).json({ error: 'OTP has expired. Please request a new one.' });
    }

    if (record.otp !== otp.trim()) {
      return res.status(400).json({ error: 'Incorrect OTP. Please check and try again.' });
    }

    console.log(`[Auth] OTP verified successfully for: ${emailLower}`);
    res.json({ success: true, message: 'OTP verified successfully.' });
  } catch (err: any) {
    console.error('Error in /api/auth/verify-otp:', err);
    res.status(500).json({ error: 'Internal server error during OTP verification.' });
  }
});

/**
 * POST /api/auth/reset-password
 * Verifies OTP once more then updates the clinician's hashed password and deletes the OTP.
 */
app.post('/api/auth/reset-password', authLimiter, async (req: Request, res: Response): Promise<any> => {
  try {
    const { email, otp, newPassword } = req.body;
    if (!email || !otp || !newPassword) {
      return res.status(400).json({ error: 'Email, OTP, and new password are required.' });
    }

    if (newPassword.length < 6) {
      return res.status(400).json({ error: 'Password must be at least 6 characters.' });
    }

    const emailLower = email.trim().toLowerCase();
    const record = await PasswordResetOtp.findOne({ email: emailLower });

    if (!record) {
      return res.status(400).json({ error: 'OTP not found or already used. Please request a new one.' });
    }

    if (new Date() > record.expiresAt) {
      await PasswordResetOtp.deleteMany({ email: emailLower });
      return res.status(400).json({ error: 'OTP has expired. Please request a new one.' });
    }

    if (record.otp !== otp.trim()) {
      return res.status(400).json({ error: 'Incorrect OTP. Cannot reset password.' });
    }

    // Hash the new password and update the user
    const hashedPassword = await bcrypt.hash(newPassword, 10);
    await User.updateOne({ email: emailLower }, { password: hashedPassword });

    // Invalidate OTP immediately after use
    await PasswordResetOtp.deleteMany({ email: emailLower });

    console.log(`[Auth] Password reset successfully for: ${emailLower}`);
    res.json({ success: true, message: 'Password reset successfully. You can now log in.' });
  } catch (err: any) {
    console.error('Error in /api/auth/reset-password:', err);
    res.status(500).json({ error: 'Internal server error during password reset.' });
  }
});

// ─── PROTECTED CLINICAL ENDPOINTS (SECURED WITH authenticateToken) ─────────────

/**
 * GET /api/dashboard
 * Dynamically computes stats, lists recent cases, and synthesizes dynamic case insights.
 * Isolated dynamically to return ONLY cases reviewed by the logged-in user ID.
 */
app.get('/api/dashboard', authenticateToken as any, async (req: AuthRequest, res: Response) => {
  try {
    const userId = req.user?.id || '';

    // Fetch all data in parallel for performance
    const [userCases, chatSessionsCount, distinctPatientIds] = await Promise.all([
      Case.find({ userId }).sort({ createdAt: -1 }),
      ChatSessionModel.countDocuments({ userId }),
      // Count unique non-empty patientIds to get true total patients
      Case.distinct('patientId', { userId, patientId: { $nin: ['', null] as string[] } }),
    ]);
    const total = userCases.length;
    // Unique patients: count distinct non-empty patientIds; fallback to total if none
    const totalPatients = distinctPatientIds.length > 0 ? distinctPatientIds.length : total;

    // Aggregate site distribution percentages dynamically
    const siteCounts: Record<string, number> = {};
    userCases.forEach((c) => {
      siteCounts[c.site] = (siteCounts[c.site] || 0) + 1;
    });
    const distribution = Object.entries(siteCounts).map(([label, count]) => ({
      label,
      pct: total > 0 ? Math.round((count / total) * 100) : 0,
    }));

    // Create a dynamic case insight highlighting the latest uploaded patient
    let insight = {
      patientId: 'N/A',
      text: 'Upload a patient CT scan or pathology report to surface clinical summaries and stage insights.',
    };
    if (total > 0) {
      const latest = userCases[0];
      insight = {
        patientId: latest.patientId,
        text: `${latest.patientId} shows ${latest.site} staging ${latest.tnm}. Review staging protocols and considerations before Thursday's MDT review.`,
      };
    }

    res.json({
      stats: [
        { label: 'Cases Reviewed', value: total.toString() },
        { label: 'Total Patients', value: totalPatients.toString() },
        { label: 'Chat Sessions', value: chatSessionsCount.toString() },
        { label: 'Avg. Processing', value: total > 0 ? '1m 18s' : '0s' },
      ],
      recent: userCases,
      insight,
      distribution: distribution.sort((a, b) => b.pct - a.pct),
    });
  } catch (error: any) {
    console.error('Error in /api/dashboard:', error);
    res.status(500).json({ error: 'Internal server error fetching dashboard.' });
  }
});

app.get('/health', async (req, res) => {
  res.json({ status: "ok" })
})

/**
 * GET /api/profile
 * Returns the current authenticated surgeon profile and dynamically computed staging summaries.
 */
app.get('/api/profile', authenticateToken as any, async (req: AuthRequest, res: Response) => {
  try {
    const userId = req.user?.id || '';
    const userCases = await Case.find({ userId });
    const total = userCases.length;

    // Find the user's account details
    const user = await User.findById(userId);
    const profile = user
      ? { name: user.name, specialty: user.specialty, institution: user.institution }
      : { name: 'Dr. Guest', specialty: 'Head & Neck Oncology Surgeon', institution: 'ScanWise Medical' };

    // Calculate most common staging dynamically
    const tnmCounts: Record<string, number> = {};
    userCases.forEach((c) => {
      tnmCounts[c.tnm] = (tnmCounts[c.tnm] || 0) + 1;
    });
    const commonTnm = Object.entries(tnmCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'N/A';

    // Calculate most common primary site dynamically
    const siteCounts: Record<string, number> = {};
    userCases.forEach((c) => {
      siteCounts[c.site] = (siteCounts[c.site] || 0) + 1;
    });
    const commonSite = Object.entries(siteCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'N/A';

    res.json({
      userProfile: profile,
      stats: [
        { l: 'Total cases', v: total.toString() },
        { l: 'Avg TNM stage', v: commonTnm },
        { l: 'Common site', v: commonSite },
      ],
    });
  } catch (error: any) {
    console.error('Error in GET /api/profile:', error);
    res.status(500).json({ error: 'Internal server error fetching profile.' });
  }
});

/**
 * POST /api/profile
 * Dynamically edits the logged-in surgeon's profile details.
 */
app.post('/api/profile', authenticateToken as any, async (req: AuthRequest, res: Response) => {
  try {
    const userId = req.user?.id || '';
    const { name, specialty, institution } = req.body;
    if (!name || !specialty || !institution) {
      return res.status(400).json({ error: 'Name, specialty, and institution are required.' });
    }

    const updatedUser = await User.findByIdAndUpdate(
      userId,
      { name, specialty, institution },
      { new: true }
    );

    if (updatedUser) {
      console.log(`[Profile] Updated profile details for: ${updatedUser.email}`);
      res.json({
        success: true,
        userProfile: {
          name: updatedUser.name,
          specialty: updatedUser.specialty,
          institution: updatedUser.institution,
        }
      });
    } else {
      res.status(404).json({ error: 'Surgeon profile not found.' });
    }
  } catch (error: any) {
    console.error('Error in POST /api/profile:', error);
    res.status(500).json({ error: 'Internal server error updating profile.' });
  }
});

/**
 * POST /api/clear-cases
 * Wipes out ONLY the cases belonging to the authenticated surgeon ID.
 */
app.post('/api/clear-cases', authenticateToken as any, async (req: AuthRequest, res: Response) => {
  try {
    const userId = req.user?.id || '';

    // Wipes references cache for this user
    for (const key of referenceCache.keys()) {
      if (key.startsWith(userId + '_')) {
        referenceCache.delete(key);
      }
    }

    await Case.deleteMany({ userId });
    console.log(`[DB] Wiped case registry for user: ${userId}`);
    res.json({ success: true, message: 'All case registry data cleared.' });
  } catch (error: any) {
    console.error('Error in /api/clear-cases:', error);
    res.status(500).json({ error: 'Internal server error clearing cases.' });
  }
});

/**
 * POST /api/upload
 * Analyzes pathology report (PDF) or CT scan metadata.
 * Saves the generated case summary prefixed with the user's ID.
 */
app.post('/api/upload', aiDailyLimiter, aiMinutlyLimiter, authenticateToken as any, (req: AuthRequest, res: Response, next: NextFunction) => {
  upload.single('file')(req as any, res as any, (err: any) => {
    if (err) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(400).json({
          error: 'File is too large. Please upload a scan or report smaller than 10MB.',
        });
      }
      return res.status(400).json({ error: err.message || 'File upload failed.' });
    }
    next();
  });
}, async (req: AuthRequest, res: Response): Promise<any> => {
  try {
    const userId = req.user?.id || '';
    const userPatientId = req.body.patientId || '';
    let rawText = '';
    let fileName = '';
    let isImage = false;
    let base64Image = '';
    let mimeType = '';

    if (req.file) {
      fileName = req.file.originalname;
      const fileBuffer = req.file.buffer;

      if (req.file.mimetype === 'application/pdf' || fileName.toLowerCase().endsWith('.pdf')) {
        console.log(`Parsing pathology PDF: ${fileName}`);
        const parsedPdf = await pdfParse(fileBuffer);
        rawText = parsedPdf.text;
      } else {
        isImage = true;
        base64Image = fileBuffer.toString('base64');
        mimeType = req.file.mimetype || 'image/jpeg';
      }
    } else {
      const { metadata } = req.body;
      rawText = metadata || `Patient Record: ${userPatientId || 'PT-2024-0041'}`;
    }

    if (!isImage && !rawText.trim()) {
      return res.status(400).json({ error: 'No report text or file content found to analyze.' });
    }

    console.log(`Generating AI clinical summary via Groq/Gemini for user ${userId}...`);

    const systemPrompt = `first analyze and tell if valid. If invalid, you MUST return a JSON object with: { "isValid": false, "error": "invalid" }

You are an expert AI clinical decision support system for head and neck oncology surgeons.
Your task is to analyze pathology reports, pathology details, or clinical scans (images) of a head/neck cancer patient and synthesize an extremely comprehensive, detailed, and high-yield structured clinical summary in JSON format.
You must output extensive, comprehensive clinical descriptions with absolute specificity, leaving out no diagnostic indicators.

Validation Check Rules:
- If analyzing text, verify if it contains relevant medical, clinical, or oncological details. If the text does not contain relevant oncological or medical details, it is invalid and you MUST return a JSON object with: { "isValid": false, "error": "invalid" }.
- If analyzing an image, verify if the image is a valid medical scan (such as a CT scan, MRI scan, X-ray, PET scan, pathology slide, or DICOM slice). If the image is NOT a medical scan (for example, if it is a photo of a pet, a person, a landscape, a building, a UI mockup, or generic drawings), it is invalid and you MUST return a JSON object with: { "isValid": false, "error": "invalid" }.

If the input is valid, you MUST return a JSON object matching this schema exactly:
{
  "isValid": true,
  "patientId": "string (extract from report or generate a realistic one like PT-2024-XXXX)",
  "confidence": 0.95, // (number: float between 0.0 and 1.0 representing your diagnostic confidence of the site and staging)
  "site": "string (detected primary site, e.g. Base of Tongue, Larynx, Oral Tongue, Tonsil, Oropharynx)",
  "findings": [
    "detailed primary tumor dimensions (e.g., 3.4 x 2.8 x 1.5 cm) and specific pathologic features (e.g., degree of keratinization, surface ulceration, exact depth of invasion (DOI) in millimeters, perineural invasion (PNI), lymphovascular invasion (LVI), bone/mandibular cortex invasion, and deep skeletal muscle infiltration)",
    "detailed lymph node findings: total nodes harvested, exact number of positive nodes, sizes of largest metastatic deposits (e.g., largest node 4.2 cm), exact neck stations/levels (levels I-V) involved, and explicit presence or absence of Extranodal Extension (ENE) / extracapsular spread",
    "detailed margin clearance status: specify closest surgical margins in millimeters (e.g., deep margin cleared by 1.8 mm, mucosal margin cleared by 4 mm) and define whether resection margins are clinically clear, close (< 5 mm), or involved",
    "comprehensive anatomical landmarks infiltration status: describe specific adjacent tissue or nerve involvements (e.g., invasion of intrinsic tongue musculature, hyoglossus muscle, lingual nerve, or submandibular gland duct)",
    "AI-generated summary. Final clinical responsibility remains with the surgeon."
  ],
  "tnm": "string (standard AJCC 8th TNM stage, e.g. T3N2bM0)",
  "differentials": [
    { "diagnosis": "string (full pathological diagnosis)", "probability": "string (e.g. Primary, Likely, Less likely)" },
    { "diagnosis": "string", "probability": "string" }
  ],
  "surgicalConsiderations": [
    "Tracheostomy requirements and clinical details: necessity, technique (e.g., temporary surgical tracheostomy vs prolonged intubation), clinical indications based on potential post-operative edema of the upper airway, and expected decannulation timeline",
    "Reconstructive surgery blueprint: anticipated tissue transfer/flap selection (e.g., Radial Forearm Free Flap (RFFF) for thin mucosal defect vs Anterolateral Thigh (ALT) free flap for bulky muscle defect vs Fibula Free Flap (FFF) for segmental mandibular bone defect) with strict clinical reasoning based on the anticipated donor and recipient site dimensions, vessel anastomosis options (e.g., facial artery to deep lingual branches), and donor-site closure methods",
    "Neck Dissection mapping: detailed selective or comprehensive neck dissection boundaries (e.g., ipsilateral comprehensive neck dissection levels I-V, contralateral selective neck dissection levels II-IV) with rationale reflecting lymphatic drainage pathways for this specific primary tumor site",
    "Airway, Nutrition, & Supportive Care plan: detailed plans for percutaneous endoscopic gastrostomy (PEG) tube placement (pre-operative prophylactic vs post-operative reactive) to support speech/swallow rehabilitation, along with swallowing therapies, aspiration precautions, and intensive post-operative airway monitoring protocols"
  ],
  "protocol": "string (specific NCCN head and neck oncology stage sub-protocol description tailored to the site and stage. Include primary surgical resection details and specific adjuvant guidelines, e.g., adjuvant radiation therapy vs concurrent cisplatin-based chemoradiotherapy if ENE or positive margins are present)",
  "prognosticFactors": [
    "molecular and viral status: detailed HPV/p16 status, EBV/LMP1 status if relevant, and p53 expression characteristics",
    "smoking and alcohol index risk profile: exact pack-years (e.g. 35 pack-years history) and quantified impact on mutational signature, biological behavior, and overall patient survival indices",
    "AJCC risk stratification: classification of patient into high-risk, intermediate-risk, or low-risk cohort based on pathological factors, ENE status, and margin status, with estimated 5-year disease-free survival (DFS) statistics"
  ],
  "multidisciplinaryRecommendations": [
    "systemic therapy regimen recommendations: specific chemotherapy combinations (e.g., concurrent cisplatin 100 mg/m2 every 21 days for 3 cycles) or induction regimens (TPF) with dose-intensity guidelines and renal/neurological clearance thresholds",
    "adjuvant radiation target volume and dosing guidelines: specify exact radiation dose delivered (e.g., 70 Gy in 35 fractions to high-risk postoperative bed vs 54-60 Gy to elective low-risk nodal stations) using IMRT or VMAT techniques",
    "supportive and preventative rehabilitation: speech-language pathology (SLP) swallowing exercises schedule, dental/extraction prophylactic guidelines before radiotherapy start, and nutritional maintenance protocols"
  ]
}

Crucial Guidelines:
1. Be highly concise, focused, and high-yield. Do NOT write overly long or verbose paragraphs. Each item in the "findings", "surgicalConsiderations", "prognosticFactors", and "multidisciplinaryRecommendations" arrays should be a concise, information-dense clinical description (around 1-2 precise sentences) containing the exact measurements, specific structures, and clear clinical justifications without unnecessary wordiness or fluff.
2. In surgicalConsiderations, provide focused surgical details (specific neck levels, flap choice, and airway management) in a concise manner.
3. Under findings, include the clinician disclaimer: "AI-generated summary. Final clinical responsibility remains with the surgeon." as the last item.
4. Do not add any introductory or concluding text, or markdown code block markers. Return ONLY the raw JSON.
5. DO NOT copy or default to the placeholder examples mentioned in the schema instructions (such as the ID "PT-2024-XXXX", dimensions like "3.4 x 2.8 x 1.5 cm", node size "4.2 cm", margins "1.8 mm / 4 mm", staging "T3N2bM0", or site "Oropharynx"). You must dynamically generate clinical values, measurements, staging, and site names that are specific and custom-synthesized for the provided patient data.`;

    let structuredSummary: any;
    let textResponse = '';

    if (isImage) {
      let geminiSuccess = false;
      const ai = getGeminiClient();
      if (ai) {
        try {
          console.log(`Analyzing image via Gemini Model (gemini-3.1-flash-lite) for user ${userId}...`);
          const geminiResponse = await ai.models.generateContent({
            model: 'gemini-3.1-flash-lite',
            contents: [
              {
                inlineData: {
                  mimeType: mimeType,
                  data: base64Image
                }
              },
              { text: systemPrompt },
              { text: `Patient report / scan details. This is an uploaded image file (${fileName}). First analyze and tell if valid. If it is NOT a valid medical scan (like a CT scan or MRI scan), you MUST return a JSON object with: { "isValid": false, "error": "invalid" }. If it is a valid scan, describe the anatomical findings, tumor dimensions, and staging details you see.${userPatientId ? `\n\nPatient ID: ${userPatientId}` : ''}` }
            ],
            config: {
              responseMimeType: "application/json"
            }
          });
          textResponse = geminiResponse.text || '';
          console.log(`Gemini response received.`);

          const cleanedResponse = textResponse.trim().toLowerCase();
          if (cleanedResponse === 'invalid' || cleanedResponse.startsWith('invalid') || cleanedResponse.includes('"invalid"')) {
            console.log(`[Validation Error] Gemini returned invalid. Upload rejected.`);
            return res.status(400).json({
              error: 'Please upload a proper clinical document or medical scan related to head and neck oncology.'
            });
          }

          structuredSummary = extractJSON(textResponse);
          geminiSuccess = true;
        } catch (geminiErr: any) {
          console.warn(`Gemini Vision analysis failed, falling back to Groq:`, geminiErr);
        }
      }

      if (!geminiSuccess) {
        console.log(`Analyzing image via Groq Vision Model (qwen/qwen3.6-27b) for user ${userId}...`);
        const dataUrl = `data:${mimeType};base64,${base64Image}`;
        const groqResponse = await groq.chat.completions.create({
          model: 'qwen/qwen3.6-27b',
          response_format: { type: "json_object" },
          messages: [
            { role: 'system', content: systemPrompt },
            {
              role: 'user',
              content: [
                {
                  type: 'text',
                  text: `Patient report / scan details. This is an uploaded image file (${fileName}). First analyze and tell if valid. If it is NOT a valid medical scan (like a CT scan or MRI scan), you MUST return a JSON object with: { "isValid": false, "error": "invalid" }. If it is a valid scan, describe the anatomical findings, tumor dimensions, and staging details you see.${userPatientId ? `\n\nPatient ID: ${userPatientId}` : ''}`
                },
                {
                  type: 'image_url',
                  image_url: {
                    url: dataUrl
                  }
                }
              ]
            }
          ] as any,
          temperature: 0.1,
        });

        textResponse = groqResponse.choices[0]?.message?.content || '';
        const cleanedResponse = textResponse.trim().toLowerCase();
        if (cleanedResponse === 'invalid' || cleanedResponse.startsWith('invalid') || cleanedResponse.includes('"invalid"')) {
          console.log(`[Validation Error] Groq returned invalid. Upload rejected.`);
          return res.status(400).json({
            error: 'Please upload a proper clinical document or medical scan related to head and neck oncology.'
          });
        }
        structuredSummary = extractJSON(textResponse);
      }
    } else {
      let geminiTextSuccess = false;
      const textPrompt = `Analyze the following patient report / imaging details. First analyze and tell if valid. If it does not contain relevant oncology medical details, you MUST return a JSON object with: { "isValid": false, "error": "invalid" }.\n\n${rawText}${userPatientId ? `\n\nPatient ID: ${userPatientId}` : ''}`;

      const ai = getGeminiClient();
      if (ai) {
        try {
          console.log(`Generating AI clinical summary via Gemini Model (gemini-3.1-flash-lite) for user ${userId}...`);
          const geminiResponse = await ai.models.generateContent({
            model: 'gemini-3.1-flash-lite',
            contents: [
              { text: systemPrompt },
              { text: textPrompt }
            ],
            config: {
              responseMimeType: "application/json"
            }
          });
          textResponse = geminiResponse.text || '';
          console.log(`Gemini response received for text/PDF report.`);

          const cleanedResponse = textResponse.trim().toLowerCase();
          if (cleanedResponse === 'invalid' || cleanedResponse.startsWith('invalid') || cleanedResponse.includes('"invalid"')) {
            console.log(`[Validation Error] Gemini returned invalid. Upload rejected.`);
            return res.status(400).json({
              error: 'Please upload a proper clinical document or medical scan related to head and neck oncology.'
            });
          }

          structuredSummary = extractJSON(textResponse);
          geminiTextSuccess = true;
        } catch (geminiErr: any) {
          console.warn(`Gemini Text analysis failed, falling back to Groq:`, geminiErr);
        }
      }

      if (!geminiTextSuccess) {
        console.log(`Generating AI clinical summary via Groq Llama-3.1-8b-instant for user ${userId}...`);
        const groqResponse = await groq.chat.completions.create({
          model: 'llama-3.1-8b-instant',
          response_format: { type: "json_object" },
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: textPrompt }
          ],
          temperature: 0.1,
        });

        textResponse = groqResponse.choices[0]?.message?.content || '';
        const cleanedResponse = textResponse.trim().toLowerCase();
        if (cleanedResponse === 'invalid' || cleanedResponse.startsWith('invalid') || cleanedResponse.includes('"invalid"')) {
          console.log(`[Validation Error] Groq returned invalid. Upload rejected.`);
          return res.status(400).json({
            error: 'Please upload a proper clinical document or medical scan related to head and neck oncology.'
          });
        }
        structuredSummary = extractJSON(textResponse);
      }
    }

    // Validate report/scan correctness before proceeding
    if (structuredSummary.isValid === false || structuredSummary.isValid === 'false') {
      console.log(`[Validation Error] Upload rejected: ${structuredSummary.error}`);
      return res.status(400).json({ error: structuredSummary.error || 'Please upload a proper clinical document or medical scan related to head and neck oncology.' });
    }

    // Bind this case context explicitly to the active surgeon's userId!
    const rawConfidence = typeof structuredSummary.confidence === 'number'
      ? structuredSummary.confidence
      : parseFloat(structuredSummary.confidence);
    const confidenceVal = isNaN(rawConfidence) ? 1.0 : rawConfidence;
    const dateVal = 'Today, ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const dynamicCase = new Case({
      ...structuredSummary,
      userId,
      confidence: confidenceVal,
      date: dateVal,
    });

    if (userPatientId) {
      // User explicitly entered a patient ID — honour it exactly
      dynamicCase.patientId = userPatientId;
    } else {
      // No user-supplied ID: the AI may generate the same value repeatedly (low temperature).
      // Guarantee uniqueness by generating a fresh ID with a timestamp-based suffix.
      const year = new Date().getFullYear();
      const suffix = Date.now().toString(36).slice(-5).toUpperCase(); // e.g. "K3X7Q"
      dynamicCase.patientId = `PT-${year}-${suffix}`;
    }

    await dynamicCase.save();

    // Invalidate the references cache for this patient ID
    const cacheKey = `${userId}_${dynamicCase.patientId}`;
    referenceCache.delete(cacheKey);
    console.log(`[Cache Invalidate] Cleared cached references for patient ${dynamicCase.patientId} (User ${userId})`);

    console.log(`AI Clinical Summary generated and saved for patient: ${dynamicCase.patientId} (User: ${userId})`);
    return res.json(dynamicCase);

  } catch (error: any) {
    console.error('Error in /api/upload:', error);
    return res.status(500).json({ error: error.message || 'Internal server error during upload analysis.' });
  }
});

/**
 * POST /api/chat
 * Answers dynamic patient queries anchored entirely on the generated Case Context.
 */
app.post('/api/chat', aiDailyLimiter, aiMinutlyLimiter, authenticateToken as any, async (req: AuthRequest, res: Response): Promise<any> => {
  try {
    const { message, history, caseContext } = req.body;

    if (!message) {
      return res.status(400).json({ error: 'Message query is required.' });
    }
    if (!caseContext) {
      return res.status(400).json({ error: 'Case context is required. Feature 2 is downstream of Feature 1.' });
    }

    console.log(`Answering chat query for patient: ${caseContext.patientId} (User: ${req.user?.id})`);

    const systemPrompt = `You are ScanWise AI, an expert oncological assistant. You are aiding a surgeon in reviewing a specific patient case.

Here is the complete clinical context of the active patient:
${JSON.stringify(caseContext, null, 2)}

Your Absolute Rules:
1. You MUST answer the surgeon's questions ONLY by anchoring on the provided patient context above.
2. Do NOT speculate or answer from general knowledge in isolation if it directly contradicts or is completely unsupported by the patient's case context.
3. If the surgeon asks general clinical guidelines/questions (e.g. general NCCN staging rules), answer them but always frame the patient's case as the primary anchor context.
4. If a question is entirely unrelated to head and neck cancer or the patient's context, politely decline to answer, stating that you only answer in the context of the active patient report.
5. Keep your responses highly professional, medically precise, clear, and concise.`;

    const chatMessages = [
      { role: 'system', content: systemPrompt }
    ];

    if (history && Array.isArray(history)) {
      history.forEach((h: { role: 'user' | 'ai'; text: string }) => {
        chatMessages.push({
          role: h.role === 'user' ? 'user' : 'assistant',
          content: h.text,
        });
      });
    }

    chatMessages.push({ role: 'user', content: message });

    const response = await groq.chat.completions.create({
      model: 'llama-3.1-8b-instant',
      messages: chatMessages as any,
      temperature: 0.2,
    });

    const reply = response.choices[0]?.message?.content || '';
    return res.json({ reply });

  } catch (error: any) {
    console.error('Error in /api/chat:', error);
    return res.status(500).json({ error: error.message || 'Internal server error during chat query.' });
  }
});

/**
 * POST /api/reference
 * Auto-populates case-specific guidelines and queries PubMed-grade research papers.
 */
app.post('/api/reference', aiDailyLimiter, aiMinutlyLimiter, authenticateToken as any, async (req: AuthRequest, res: Response): Promise<any> => {
  try {
    const userId = req.user?.id || '';
    const { caseContext } = req.body;

    if (!caseContext) {
      return res.status(400).json({ error: 'Case context is required to surface references.' });
    }

    // Check if references are already cached for this case & user
    const cacheKey = `${userId}_${caseContext.patientId}`;
    if (referenceCache.has(cacheKey)) {
      console.log(`[Cache Hit] Returning cached reference details for key: ${cacheKey}`);
      return res.json(referenceCache.get(cacheKey));
    }

    console.log(`Generating case-specific references for key ${cacheKey}: ${caseContext.site} (${caseContext.tnm})`);

    const systemPrompt = `You are an expert AI medical reference system for head and neck oncological surgeons.
You are given the following structured patient case:
${JSON.stringify(caseContext, null, 2)}

Your task is to return a JSON object containing case-specific guidelines and scientific research paper recommendations:
- "protocols": 4-6 specific NCCN/ASCO/ESMO sub-protocol bullet points tailored EXACTLY to this primary site (${caseContext.site}) and staging (${caseContext.tnm}). Do not return generic guidelines, return exact stage sub-protocol items.
- "papers": 4-6 curated highly realistic recent (2020-2026) scientific research papers relevant to this specific staging, site, and procedure.
For each paper, you must provide:
  - "title": Highly realistic clinical research paper title (matching site and staging/treatment)
  - "authors": Main authors (e.g. Chen L, Patel R, et al.)
  - "journal": Journal name and year (e.g. JAMA Otolaryngology, 2024 or Oral Oncology, 2023)
  - "snippet": High-yield clinical summary of the paper's key finding / conclusion
  - "tag": One of 'Staging', 'Surgical technique', 'Outcomes', 'Reconstruction'
  - "cites": Realistic citation count (number)
  - "url": Highly realistic or valid PubMed/clinical URL (e.g. "https://pubmed.ncbi.nlm.nih.gov/38265432/")

Your output MUST be a valid JSON object matching this schema exactly:
{
  "protocols": [
    "specific protocol point 1",
    "specific protocol point 2"
  ],
  "papers": [
    {
      "title": "string",
      "authors": "string",
      "journal": "string",
      "snippet": "string",
      "tag": "Staging | Surgical technique | Outcomes | Reconstruction",
      "cites": 120,
      "url": "string"
    }
  ]
}

Ensure the output is 100% valid JSON. Do not add markdown backticks or other text.`;

    const response = await groq.chat.completions.create({
      model: 'llama-3.1-8b-instant',
      response_format: { type: "json_object" },
      messages: [{ role: 'user', content: systemPrompt }],
      temperature: 0.2,
    });

    const textResponse = response.choices[0]?.message?.content || '';
    const structuredReference = extractJSON(textResponse);

    // Save to user-isolated server-side cache
    referenceCache.set(cacheKey, structuredReference);
    console.log(`[Cache Populate] References cached for key: ${cacheKey}`);

    return res.json(structuredReference);

  } catch (error: any) {
    console.error('Error in /api/reference:', error);
    return res.status(500).json({ error: error.message || 'Internal server error during reference synthesis.' });
  }
});

// ─── SAVED CASES ENDPOINTS ───────────────────────────────────────────────────

/**
 * GET /api/saved-cases
 * Returns all cases explicitly bookmarked by the authenticated surgeon.
 */
app.get('/api/saved-cases', authenticateToken as any, async (req: AuthRequest, res: Response) => {
  try {
    const userId = req.user?.id || '';
    const savedCases = await SavedCase.find({ userId }).sort({ createdAt: -1 });
    return res.json({ savedCases });
  } catch (error: any) {
    console.error('Error in GET /api/saved-cases:', error);
    return res.status(500).json({ error: 'Internal server error fetching saved cases.' });
  }
});

/**
 * POST /api/saved-cases
 * Saves or replaces a single case bookmark for the authenticated surgeon.
 */
app.post('/api/saved-cases', authenticateToken as any, async (req: AuthRequest, res: Response) => {
  try {
    const userId = req.user?.id || '';
    const caseData = req.body;

    if (!caseData || !caseData.patientId || !caseData.site || !caseData.tnm) {
      return res.status(400).json({ error: 'patientId, site, and tnm are required.' });
    }

    // Upsert — update if already saved, insert if not
    await SavedCase.findOneAndUpdate(
      { userId, patientId: caseData.patientId },
      { ...caseData, userId },
      { upsert: true, new: true }
    );

    console.log(`[SavedCases] Saved case ${caseData.patientId} for user ${userId}`);
    return res.json({ success: true });
  } catch (error: any) {
    console.error('Error in POST /api/saved-cases:', error);
    return res.status(500).json({ error: 'Internal server error saving case.' });
  }
});

/**
 * PUT /api/saved-cases/sync
 * Replaces all saved cases for the authenticated surgeon with the provided list.
 */
app.put('/api/saved-cases/sync', authenticateToken as any, async (req: AuthRequest, res: Response) => {
  try {
    const userId = req.user?.id || '';
    const { savedCases } = req.body;

    if (!Array.isArray(savedCases)) {
      return res.status(400).json({ error: 'savedCases must be an array.' });
    }

    // Delete all existing, insert fresh list
    await SavedCase.deleteMany({ userId });
    if (savedCases.length > 0) {
      await SavedCase.insertMany(savedCases.map((c: any) => ({ ...c, userId })));
    }

    console.log(`[SavedCases] Synced ${savedCases.length} saved cases for user ${userId}`);
    return res.json({ success: true });
  } catch (error: any) {
    console.error('Error in PUT /api/saved-cases/sync:', error);
    return res.status(500).json({ error: 'Internal server error syncing saved cases.' });
  }
});

/**
 * DELETE /api/saved-cases/:patientId
 * Removes a single bookmarked case for the authenticated surgeon.
 */
app.delete('/api/saved-cases/:patientId', authenticateToken as any, async (req: AuthRequest, res: Response) => {
  try {
    const userId = req.user?.id || '';
    const { patientId } = req.params;
    await SavedCase.deleteOne({ userId, patientId });
    console.log(`[SavedCases] Deleted case ${patientId} for user ${userId}`);
    return res.json({ success: true });
  } catch (error: any) {
    console.error('Error in DELETE /api/saved-cases:', error);
    return res.status(500).json({ error: 'Internal server error deleting saved case.' });
  }
});

// ─── CHAT SESSION ENDPOINTS ───────────────────────────────────────────────────

/**
 * GET /api/chat-sessions
 * Returns all chat sessions for the authenticated surgeon.
 */
app.get('/api/chat-sessions', authenticateToken as any, async (req: AuthRequest, res: Response) => {
  try {
    const userId = req.user?.id || '';
    const sessions = await ChatSessionModel.find({ userId }).sort({ updatedAt: -1 });
    return res.json({ chatSessions: sessions });
  } catch (error: any) {
    console.error('Error in GET /api/chat-sessions:', error);
    return res.status(500).json({ error: 'Internal server error fetching chat sessions.' });
  }
});

/**
 * PUT /api/chat-sessions/sync
 * Replaces all chat sessions for the authenticated surgeon with the provided list.
 */
app.put('/api/chat-sessions/sync', authenticateToken as any, async (req: AuthRequest, res: Response) => {
  try {
    const userId = req.user?.id || '';
    const { chatSessions } = req.body;

    if (!Array.isArray(chatSessions)) {
      return res.status(400).json({ error: 'chatSessions must be an array.' });
    }

    // Upsert each session by its stable client-side sessionId
    const ops = chatSessions.map((s: any) => ({
      updateOne: {
        filter: { userId, sessionId: s.id },
        update: {
          userId,
          sessionId: s.id,
          patientId: s.patientId,
          title: s.title,
          messages: s.messages,
          caseContext: s.caseContext,
          date: s.date,
        },
        upsert: true,
      },
    }));

    if (ops.length > 0) {
      await ChatSessionModel.bulkWrite(ops);
    }

    // Remove sessions that were deleted on the client
    const clientSessionIds = chatSessions.map((s: any) => s.id);
    await ChatSessionModel.deleteMany({ userId, sessionId: { $nin: clientSessionIds } });

    console.log(`[ChatSessions] Synced ${chatSessions.length} sessions for user ${userId}`);
    return res.json({ success: true });
  } catch (error: any) {
    console.error('Error in PUT /api/chat-sessions/sync:', error);
    return res.status(500).json({ error: 'Internal server error syncing chat sessions.' });
  }
});

/**
 * DELETE /api/chat-sessions/:sessionId
 * Deletes a single chat session.
 */
app.delete('/api/chat-sessions/:sessionId', authenticateToken as any, async (req: AuthRequest, res: Response) => {
  try {
    const userId = req.user?.id || '';
    const { sessionId } = req.params;
    await ChatSessionModel.deleteOne({ userId, sessionId });
    console.log(`[ChatSessions] Deleted session ${sessionId} for user ${userId}`);
    return res.json({ success: true });
  } catch (error: any) {
    console.error('Error in DELETE /api/chat-sessions:', error);
    return res.status(500).json({ error: 'Internal server error deleting chat session.' });
  }
});

// Start the server
app.listen(port, () => {
  const geminiAvailable = !!(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY.trim());
  console.log(`====================================================`);
  console.log(`ScanWise AI backend server running on port ${port}`);
  console.log(`[AI Configuration] Primary Vision: ${geminiAvailable ? 'Gemini (gemini-3.1-flash-lite) [Fallback: Groq qwen/qwen3.6-27b]' : 'Groq Fallback (qwen/qwen3.6-27b)'}`);
  console.log(`[AI Configuration] Primary Text: ${geminiAvailable ? 'Gemini (gemini-3.1-flash-lite) [Fallback: Groq llama-3.1-8b-instant]' : 'Groq Fallback (llama-3.1-8b-instant)'}`);
  console.log(`====================================================`);
});
