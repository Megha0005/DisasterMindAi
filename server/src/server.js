import express from "express";
import cors from "cors";
import helmet from "helmet";
import dotenv from "dotenv";
import aiRoutes from "./routes/aiRoutes.js";
import authRoutes from "./routes/authRoutes.js";
import connectDB from "./config/db.js";

dotenv.config();

const app = express();

app.use(helmet());

app.use(cors({
  origin: process.env.CLIENT_URL || "http://localhost:5173"
}));

app.use(express.json({ limit: "1mb" }));

app.get("/api/health", (_req, res) => {
  res.json({
    status: "ok",
    service: "DisasterMind API"
  });
});

// Authentication routes
app.use("/api/auth", authRoutes);

// AI Engine Gateway
app.use("/api/ai", aiRoutes);

const PORT = process.env.PORT || 5000;

const startServer = async () => {
  try {
    await connectDB();

    app.listen(PORT, () => {
      console.log(`DisasterMind API running on port ${PORT}`);
      console.log("Authentication routes: /api/auth");
      console.log(
        `AI Engine URL: ${
          process.env.AI_ENGINE_URL || "http://127.0.0.1:8000"
        }`
      );
    });
  } catch (error) {
    console.error("Failed to start server:", error.message);
    process.exit(1);
  }
};

startServer();
