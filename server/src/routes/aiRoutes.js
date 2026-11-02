import express from "express";

const router = express.Router();

const AI_ENGINE_URL = (
  process.env.AI_ENGINE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

async function forwardToAI(req, res, endpoint, method = "GET") {
  try {
    const options = {
      method,
      headers: {
        "Content-Type": "application/json"
      },
      signal: AbortSignal.timeout(30000)
    };

    if (method !== "GET") {
      options.body = JSON.stringify(req.body);
    }

    const response = await fetch(
      `${AI_ENGINE_URL}${endpoint}`,
      options
    );

    const result = await response.json();

    return res.status(response.status).json(result);

  } catch (error) {
    console.error("AI Engine Gateway Error:", error.message);

    return res.status(
      error.name === "TimeoutError" ? 504 : 502
    ).json({
      error: "AI Engine is currently unavailable.",
      message: error.name === "TimeoutError"
        ? "AI Engine request timed out."
        : error.message
    });
  }
}

// AI Engine health
router.get("/health", (req, res) =>
  forwardToAI(req, res, "/health")
);

// GPU information
router.get("/gpu", (req, res) =>
  forwardToAI(req, res, "/gpu")
);

// Earthquake prediction
router.post("/predictions/earthquake", (req, res) =>
  forwardToAI(
    req,
    res,
    "/api/v1/predictions/earthquake",
    "POST"
  )
);

// Synthetic count generation
router.post("/generator/counts", (req, res) =>
  forwardToAI(
    req,
    res,
    "/api/v1/generator/counts",
    "POST"
  )
);

// Disaster scenario generation
router.post("/scenarios/generate", (req, res) =>
  forwardToAI(
    req,
    res,
    "/api/v1/scenarios/generate",
    "POST"
  )
);

export default router;
