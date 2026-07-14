import mongoose, { Schema, Document } from 'mongoose';

// ─── USER SCHEMA ──────────────────────────────────────────────────────────
export interface IUser extends Document {
  name: string;
  email: string;
  password: string;
  specialty: string;
  institution: string;
}

const UserSchema: Schema = new Schema({
  name: { type: String, required: true, trim: true },
  email: { type: String, required: true, unique: true, lowercase: true, trim: true },
  password: { type: String, required: true },
  specialty: { type: String, required: true, trim: true },
  institution: { type: String, required: true, trim: true }
}, { timestamps: true });

export const User = mongoose.model<IUser>('User', UserSchema);

// ─── CASE SCHEMA ──────────────────────────────────────────────────────────
export interface IDifferential {
  diagnosis: string;
  probability: string;
}

export interface ICase extends Document {
  patientId: string;
  site: string;
  findings: string[];
  tnm: string;
  differentials: IDifferential[];
  surgicalConsiderations: string[];
  protocol: string;
  prognosticFactors: string[];
  multidisciplinaryRecommendations: string[];
  userId: string;
  confidence: number;
  date: string;
}

const CaseSchema: Schema = new Schema({
  patientId: { type: String, required: true, trim: true },
  site: { type: String, required: true, trim: true },
  findings: [{ type: String }],
  tnm: { type: String, required: true, trim: true },
  differentials: [{
    diagnosis: { type: String, required: true },
    probability: { type: String, required: true }
  }],
  surgicalConsiderations: [{ type: String }],
  protocol: { type: String },
  prognosticFactors: [{ type: String }],
  multidisciplinaryRecommendations: [{ type: String }],
  userId: { type: String, required: true },
  confidence: { type: Number, required: true },
  date: { type: String, required: true }
}, { timestamps: true });

CaseSchema.index({ userId: 1, createdAt: -1 });
export const Case = mongoose.model<ICase>('Case', CaseSchema);

// ─── SAVED CASE SCHEMA ────────────────────────────────────────────────────
// Represents the subset of cases a surgeon explicitly bookmarks.
export interface ISavedCase extends Document {
  userId: string;
  patientId: string;
  site: string;
  tnm: string;
  confidence?: number;
  findings?: string[];
  differentials?: { diagnosis: string; probability: string }[];
  surgicalConsiderations?: string[];
  protocol?: string;
  prognosticFactors?: string[];
  multidisciplinaryRecommendations?: string[];
  date?: string;
}

const SavedCaseSchema: Schema = new Schema({
  userId: { type: String, required: true },
  patientId: { type: String, required: true },
  site: { type: String, required: true },
  tnm: { type: String, required: true },
  confidence: { type: Number },
  findings: [{ type: String }],
  differentials: [{ diagnosis: String, probability: String }],
  surgicalConsiderations: [{ type: String }],
  protocol: { type: String },
  prognosticFactors: [{ type: String }],
  multidisciplinaryRecommendations: [{ type: String }],
  date: { type: String }
}, { timestamps: true });

SavedCaseSchema.index({ userId: 1 });
export const SavedCase = mongoose.model<ISavedCase>('SavedCase', SavedCaseSchema);

// ─── CHAT SESSION SCHEMA ──────────────────────────────────────────────────
export interface IChatMessage {
  role: 'user' | 'ai';
  text: string;
  t: string;
}

export interface IChatSession extends Document {
  userId: string;
  sessionId: string;  // Client-generated ID for stable references
  patientId: string;
  title: string;
  messages: IChatMessage[];
  caseContext: Record<string, any>;
  date: string;
}

const ChatMessageSchema = new Schema({
  role: { type: String, enum: ['user', 'ai'], required: true },
  text: { type: String, required: true },
  t: { type: String, required: true }
}, { _id: false });

const ChatSessionSchema: Schema = new Schema({
  userId: { type: String, required: true },
  sessionId: { type: String, required: true },  // Client-generated stable ID
  patientId: { type: String, required: true },
  title: { type: String, required: true },
  messages: [ChatMessageSchema],
  caseContext: { type: Schema.Types.Mixed, required: true },
  date: { type: String, required: true }
}, { timestamps: true });

ChatSessionSchema.index({ userId: 1, updatedAt: -1 });
ChatSessionSchema.index({ userId: 1, sessionId: 1 }, { unique: true });
export const ChatSession = mongoose.model<IChatSession>('ChatSession', ChatSessionSchema);

// ─── PASSWORD RESET OTP SCHEMA ────────────────────────────────────────────
export interface IPasswordResetOtp extends Document {
  email: string;
  otp: string;
  expiresAt: Date;
}

const PasswordResetOtpSchema: Schema = new Schema({
  email: { type: String, required: true, lowercase: true, trim: true },
  otp: { type: String, required: true },
  expiresAt: { type: Date, required: true, index: { expires: 0 } }, // TTL index — auto-deletes after expiry
});

PasswordResetOtpSchema.index({ email: 1 });
export const PasswordResetOtp = mongoose.model<IPasswordResetOtp>('PasswordResetOtp', PasswordResetOtpSchema);

// ─── DATABASE INITIALIZATION ──────────────────────────────────────────────
export async function connectDatabase() {
  const uri = process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/ocnodetect';
  try {
    await mongoose.connect(uri);
    console.log("[MongoDB] Connected successfully to DB");
    startMongoKeepAlive();
  } catch (error) {
    console.error('[MongoDB] Connection failed:', error);
    process.exit(1);
  }
}

function startMongoKeepAlive() {
  // Ping the database periodically to prevent the cluster from being paused due to inactivity.
  const intervalMinutes = parseInt(process.env.MONGODB_KEEP_ALIVE_INTERVAL || '10', 10);
  const intervalMs = intervalMinutes * 60 * 1000;

  console.log(`[MongoDB Keep-Alive] Starting keep-alive ping loop every ${intervalMinutes} minutes`);

  setInterval(async () => {
    try {
      if (mongoose.connection.readyState === 1) {
        const db = mongoose.connection.db;
        if (db) {
          await db.command({ ping: 1 });
          console.log(`[MongoDB Keep-Alive] Pinged MongoDB cluster successfully at ${new Date().toISOString()}`);
        }
      } else {
        console.warn(`[MongoDB Keep-Alive] Skip ping: readyState is ${mongoose.connection.readyState} (not connected)`);
      }
    } catch (err) {
      console.error('[MongoDB Keep-Alive] Failed to ping MongoDB:', err);
    }
  }, intervalMs);
}

