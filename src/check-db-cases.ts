import mongoose from 'mongoose';
import { connectDatabase, Case } from './database';

async function run() {
  await connectDatabase();
  console.log("Connected to MongoDB.");

  const cases = await Case.find({});
  console.log(`Total cases found in DB: ${cases.length}`);
  cases.forEach((c, i) => {
    console.log(`Case ${i+1}: patientId=${c.patientId}, site=${c.site}, tnm=${c.tnm}`);
  });

  await mongoose.disconnect();
}

run().catch(console.error);
