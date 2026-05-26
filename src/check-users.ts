import mongoose from 'mongoose';
import { connectDatabase, User } from './database';

async function run() {
  await mongoose.connect("mongodb+srv://akashkrish1010_db_user:ak123%40@cluster0.phixjoa.mongodb.net/");
  console.log("Connected to MongoDB.");

  const users = await User.find({});
  console.log(`Total users found: ${users.length}`);
  users.forEach((u, i) => {
    console.log(`User ${i+1}: id=${u._id}, email=${u.email}, name=${u.name}`);
  });

  await mongoose.disconnect();
}

run().catch(console.error);
