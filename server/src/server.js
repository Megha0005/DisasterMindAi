import express from "express";
import cors from "cors";
import helmet from "helmet";
import dotenv from "dotenv";
import aiRoutes from "./routes/aiRoutes.js";

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

// AI Engine Gateway
app.use("/api/ai", aiRoutes);

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`DisasterMind API running on port ${PORT}`);
  console.log(
    `AI Engine URL: ${
      process.env.AI_ENGINE_URL || "http://127.0.0.1:8000"
    }`
  );
});
