import express from 'express';
import http from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import mongoose from 'mongoose';
import Redis from 'redis';
import cron from 'node-cron';
import yahooFinance from 'yahoo-finance2';
import Alpha from 'alpha-vantage';
import WebSocket from 'ws';
import dotenv from 'dotenv';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import validator from 'validator';
import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import moment from 'moment';
import _ from 'lodash';

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Initialize Express app
const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: process.env.CLIENT_URL || "http://localhost:5173",
    methods: ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    credentials: true,
    allowedHeaders: ["Content-Type", "Authorization", "X-Requested-With"]
  },
  transports: ['websocket', 'polling'],
  allowEIO3: true
});

// Redis client setup with retry logic
const redisClient = Redis.createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379',
  retry_strategy: (options) => {
    if (options.error && options.error.code === 'ECONNREFUSED') {
      return new Error('Redis server connection refused');
    }
    if (options.total_retry_time > 1000 * 60 * 60) {
      return new Error('Retry time exhausted');
    }
    if (options.attempt > 10) {
      return undefined;
    }
    return Math.min(options.attempt * 100, 3000);
  }
});

redisClient.on('error', (err) => {
  console.error('Redis Client Error:', err);
});

redisClient.on('connect', () => {
  console.log('Redis Client Connected');
});

redisClient.on('ready', () => {
  console.log('Redis Client Ready');
});

redisClient.on('end', () => {
  console.log('Redis Client Connection Ended');
});

// Connect to Redis
await redisClient.connect().catch(err => {
  console.error('Redis connection failed:', err);
});

// Alpha Vantage API setup
const alpha = Alpha({ key: process.env.ALPHA_VANTAGE_API_KEY });

// Advanced middleware setup
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
      scriptSrc: ["'self'", "'unsafe-eval'"],
      fontSrc: ["'self'", "https://fonts.gstatic.com"],
      imgSrc: ["'self'", "data:", "https:", "blob:"],
      connectSrc: ["'self'", "ws:", "wss:", "https:"],
      mediaSrc: ["'self'"],
      objectSrc: ["'none'"],
      childSrc: ["'self'"],
      frameSrc: ["'self'"],
      workerSrc: ["'self'"],
      manifestSrc: ["'self'"]
    }
  },
  crossOriginEmbedderPolicy: false,
  crossOriginResourcePolicy: { policy: "cross-origin" }
}));

app.use(compression({
  level: 6,
  threshold: 1024,
  filter: (req, res) => {
    if (req.headers['x-no-compression']) {
      return false;
    }
    return compression.filter(req, res);
  }
}));

app.use(cors({
  origin: (origin, callback) => {
    const allowedOrigins = [
      process.env.CLIENT_URL,
      'http://localhost:5173',
      'http://localhost:3000',
      'https://localhost:5173'
    ].filter(Boolean);
    
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With', 'Accept', 'Origin'],
  exposedHeaders: ['X-Total-Count', 'X-Page-Count', 'Link']
}));

// Advanced rate limiting with different tiers
const createRateLimiter = (windowMs, max, message) => {
  return rateLimit({
    windowMs,
    max,
    message: { error: message },
    standardHeaders: true,
    legacyHeaders: false,
    handler: (req, res) => {
      res.status(429).json({
        error: message,
        retryAfter: Math.round(windowMs / 1000)
      });
    },
    skip: (req) => {
      // Skip rate limiting for health checks
      return req.path === '/health' || req.path === '/api/health';
    }
  });
};

// Different rate limits for different endpoints
const generalLimiter = createRateLimiter(15 * 60 * 1000, 1000, 'Too many requests from this IP');
const authLimiter = createRateLimiter(15 * 60 * 1000, 10, 'Too many authentication attempts');
const predictionLimiter = createRateLimiter(60 * 1000, 30, 'Too many prediction requests');
const dataLimiter = createRateLimiter(60 * 1000, 100, 'Too many data requests');

app.use('/api', generalLimiter);
app.use('/api/auth', authLimiter);
app.use('/api/predictions', predictionLimiter);
app.use('/api/stocks', dataLimiter);

// Body parsing middleware with size limits
app.use(express.json({
  limit: '10mb',
  verify: (req, res, buf) => {
    try {
      JSON.parse(buf);
    } catch (e) {
      res.status(400).json({ error: 'Invalid JSON format' });
      return;
    }
  }
}));

app.use(express.urlencoded({
  extended: true,
  limit: '10mb',
  parameterLimit: 1000
}));

// Request logging middleware
app.use((req, res, next) => {
  const start = Date.now();
  const originalSend = res.send;
  
  res.send = function(data) {
    const duration = Date.now() - start;
    console.log(`${req.method} ${req.path} - ${res.statusCode} - ${duration}ms`);
    originalSend.call(this, data);
  };
  
  next();
});

// MongoDB connection with advanced options
const mongoOptions = {
  useNewUrlParser: true,
  useUnifiedTopology: true,
  maxPoolSize: 20,
  minPoolSize: 5,
  serverSelectionTimeoutMS: 5000,
  socketTimeoutMS: 45000,
  family: 4,
  bufferMaxEntries: 0,
  retryWrites: true,
  retryReads: true,
  readPreference: 'primary',
  writeConcern: {
    w: 'majority',
    j: true,
    wtimeout: 10000
  }
};

mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/stockpredictor', mongoOptions);

// MongoDB event handlers
mongoose.connection.on('connected', () => {
  console.log('MongoDB connected successfully');
});

mongoose.connection.on('error', (err) => {
  console.error('MongoDB connection error:', err);
  process.exit(1);
});

mongoose.connection.on('disconnected', () => {
  console.log('MongoDB disconnected');
});

mongoose.connection.on('reconnected', () => {
  console.log('MongoDB reconnected');
});

// Graceful shutdown handling
process.on('SIGINT', async () => {
  console.log('Received SIGINT, closing MongoDB connection...');
  await mongoose.connection.close();
  process.exit(0);
});

// Enhanced database schemas
const StockSchema = new mongoose.Schema({
  symbol: {
    type: String,
    required: true,
    unique: true,
    uppercase: true,
    trim: true,
    maxlength: 10
  },
  name: {
    type: String,
    required: true,
    trim: true,
    maxlength: 200
  },
  exchange: {
    type: String,
    required: true,
    uppercase: true
  },
  sector: {
    type: String,
    trim: true
  },
  industry: {
    type: String,
    trim: true
  },
  marketCap: {
    type: Number,
    min: 0
  },
  price: {
    type: Number,
    required: true,
    min: 0
  },
  volume: {
    type: Number,
    min: 0
  },
  avgVolume: {
    type: Number,
    min: 0
  },
  beta: Number,
  eps: Number,
  pe: Number,
  dividendYield: {
    type: Number,
    min: 0,
    max: 100
  },
  fiftyTwoWeekHigh: Number,
  fiftyTwoWeekLow: Number,
  dayHigh: Number,
  dayLow: Number,
  openPrice: Number,
  previousClose: Number,
  change: Number,
  changePercent: Number,
  priceToBook: Number,
  priceToSales: Number,
  profitMargin: Number,
  operatingMargin: Number,
  returnOnAssets: Number,
  returnOnEquity: Number,
  debtToEquity: Number,
  currentRatio: Number,
  quickRatio: Number,
  revenueGrowth: Number,
  earningsGrowth: Number,
  analystRating: {
    type: String,
    enum: ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
  },
  priceTarget: Number,
  lastUpdated: {
    type: Date,
    default: Date.now
  },
  isActive: {
    type: Boolean,
    default: true
  },
  metadata: {
    dataSource: {
      type: String,
      enum: ['yahoo', 'alpha-vantage', 'polygon', 'manual']
    },
    lastFetch: Date,
    fetchCount: {
      type: Number,
      default: 0
    },
    reliability: {
      type: Number,
      min: 0,
      max: 1,
      default: 1
    }
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Add indexes for performance
StockSchema.index({ symbol: 1 });
StockSchema.index({ sector: 1 });
StockSchema.index({ marketCap: -1 });
StockSchema.index({ volume: -1 });
StockSchema.index({ lastUpdated: -1 });
StockSchema.index({ 'metadata.dataSource': 1 });

// Virtual fields
StockSchema.virtual('marketCapFormatted').get(function() {
  if (!this.marketCap) return 'N/A';
  const billion = 1000000000;
  const million = 1000000;
  
  if (this.marketCap >= billion) {
    return `$${(this.marketCap / billion).toFixed(2)}B`;
  } else if (this.marketCap >= million) {
    return `$${(this.marketCap / million).toFixed(2)}M`;
  } else {
    return `$${this.marketCap.toLocaleString()}`;
  }
});

StockSchema.virtual('priceChangeFormatted').get(function() {
  if (this.change === undefined || this.changePercent === undefined) return 'N/A';
  const sign = this.change >= 0 ? '+' : '';
  return `${sign}$${this.change.toFixed(2)} (${sign}${this.changePercent.toFixed(2)}%)`;
});

// Enhanced prediction schema
const PredictionSchema = new mongoose.Schema({
  symbol: {
    type: String,
    required: true,
    uppercase: true,
    trim: true
  },
  timeframe: {
    type: String,
    required: true,
    enum: ['1H', '4H', '1D', '1W', '1M', '3M', '6M', '1Y']
  },
  currentPrice: {
    type: Number,
    required: true,
    min: 0
  },
  predictedPrice: {
    type: Number,
    required: true,
    min: 0
  },
  priceRange: {
    low: {
      type: Number,
      required: true,
      min: 0
    },
    high: {
      type: Number,
      required: true,
      min: 0
    }
  },
  confidence: {
    type: Number,
    required: true,
    min: 0,
    max: 100
  },
  direction: {
    type: String,
    enum: ['bullish', 'bearish', 'neutral'],
    required: true
  },
  probability: {
    bullish: {
      type: Number,
      min: 0,
      max: 1
    },
    bearish: {
      type: Number,
      min: 0,
      max: 1
    },
    neutral: {
      type: Number,
      min: 0,
      max: 1
    }
  },
  keyFactors: [{
    name: {
      type: String,
      required: true
    },
    weight: {
      type: Number,
      min: 0,
      max: 1
    },
    impact: {
      type: String,
      enum: ['positive', 'negative', 'neutral']
    },
    description: String
  }],
  technicalIndicators: {
    rsi: {
      value: Number,
      signal: {
        type: String,
        enum: ['overbought', 'oversold', 'neutral']
      }
    },
    macd: {
      macd: Number,
      signal: Number,
      histogram: Number,
      trend: {
        type: String,
        enum: ['bullish', 'bearish', 'neutral']
      }
    },
    bollingerBands: {
      upper: Number,
      middle: Number,
      lower: Number,
      position: {
        type: String,
        enum: ['above_upper', 'between', 'below_lower']
      }
    },
    movingAverages: {
      sma20: Number,
      sma50: Number,
      sma200: Number,
      ema12: Number,
      ema26: Number,
      trend: {
        type: String,
        enum: ['uptrend', 'downtrend', 'sideways']
      }
    },
    momentum: {
      stochastic: {
        k: Number,
        d: Number,
        signal: String
      },
      williamsR: Number,
      roc: Number,
      momentum: Number
    },
    volume: {
      obv: Number,
      volumeTrend: {
        type: String,
        enum: ['increasing', 'decreasing', 'stable']
      },
      volumeRatio: Number
    }
  },
  fundamentalMetrics: {
    valuation: {
      peRatio: Number,
      pegRatio: Number,
      priceToBook: Number,
      priceToSales: Number,
      enterpriseValue: Number,
      evToEbitda: Number
    },
    profitability: {
      profitMargin: Number,
      operatingMargin: Number,
      grossMargin: Number,
      returnOnAssets: Number,
      returnOnEquity: Number,
      returnOnInvestment: Number
    },
    financial: {
      debtToEquity: Number,
      currentRatio: Number,
      quickRatio: Number,
      cashRatio: Number,
      interestCoverage: Number
    },
    growth: {
      revenueGrowth: Number,
      earningsGrowth: Number,
      bookValueGrowth: Number,
      dividendGrowth: Number
    }
  },
  marketSentiment: {
    overall: {
      type: String,
      enum: ['very_bullish', 'bullish', 'neutral', 'bearish', 'very_bearish']
    },
    newsScore: {
      type: Number,
      min: -1,
      max: 1
    },
    socialScore: {
      type: Number,
      min: -1,
      max: 1
    },
    analystRating: {
      consensus: String,
      priceTarget: Number,
      priceTargetRange: {
        low: Number,
        high: Number
      }
    },
    institutionalActivity: {
      ownership: Number,
      recentChanges: String,
      insiderTrading: String
    }
  },
  riskAssessment: {
    overall: {
      type: String,
      enum: ['very_low', 'low', 'moderate', 'high', 'very_high']
    },
    volatility: {
      historical: Number,
      implied: Number,
      percentile: Number
    },
    beta: Number,
    maxDrawdown: Number,
    valueAtRisk: {
      oneDay: Number,
      oneWeek: Number,
      oneMonth: Number
    },
    factors: [{
      category: {
        type: String,
        enum: ['market', 'sector', 'company', 'regulatory', 'economic']
      },
      description: String,
      severity: {
        type: String,
        enum: ['low', 'medium', 'high', 'critical']
      },
      probability: {
        type: Number,
        min: 0,
        max: 1
      }
    }]
  },
  modelPerformance: {
    algorithm: {
      type: String,
      required: true,
      enum: ['lstm', 'random_forest', 'arima', 'ensemble', 'transformer']
    },
    version: {
      type: String,
      required: true
    },
    accuracy: {
      type: Number,
      min: 0,
      max: 1
    },
    precision: {
      type: Number,
      min: 0,
      max: 1
    },
    recall: {
      type: Number,
      min: 0,
      max: 1
    },
    f1Score: {
      type: Number,
      min: 0,
      max: 1
    },
    backtestResults: {
      totalReturn: Number,
      sharpeRatio: Number,
      maxDrawdown: Number,
      winRate: Number,
      avgWin: Number,
      avgLoss: Number,
      profitFactor: Number
    },
    trainingData: {
      startDate: Date,
      endDate: Date,
      samples: Number,
      features: [String]
    },
    lastTrained: Date,
    validationScore: Number
  },
  prediction_id: {
    type: String,
    unique: true,
    required: true
  },
  status: {
    type: String,
    enum: ['active', 'expired', 'validated', 'cancelled'],
    default: 'active'
  },
  expiresAt: {
    type: Date,
    required: true
  },
  actualOutcome: {
    price: Number,
    direction: String,
    accuracy: Number,
    validatedAt: Date
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes for predictions
PredictionSchema.index({ symbol: 1, timeframe: 1, createdAt: -1 });
PredictionSchema.index({ expiresAt: 1 }, { expireAfterSeconds: 0 });
PredictionSchema.index({ prediction_id: 1 });
PredictionSchema.index({ status: 1 });
PredictionSchema.index({ 'modelPerformance.algorithm': 1 });

// Virtual fields for predictions
PredictionSchema.virtual('expectedReturn').get(function() {
  if (!this.currentPrice || !this.predictedPrice) return 0;
  return ((this.predictedPrice - this.currentPrice) / this.currentPrice) * 100;
});

PredictionSchema.virtual('riskRewardRatio').get(function() {
  if (!this.priceRange || !this.currentPrice) return 0;
  const potentialGain = Math.abs(this.predictedPrice - this.currentPrice);
  const potentialLoss = Math.abs(this.currentPrice - this.priceRange.low);
  return potentialLoss > 0 ? potentialGain / potentialLoss : 0;
});

// User schema with enhanced features
const UserSchema = new mongoose.Schema({
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true,
    validate: [validator.isEmail, 'Invalid email format']
  },
  password: {
    type: String,
    required: true,
    minlength: 8
  },
  profile: {
    firstName: {
      type: String,
      trim: true,
      maxlength: 50
    },
    lastName: {
      type: String,
      trim: true,
      maxlength: 50
    },
    phone: {
      type: String,
      validate: {
        validator: function(v) {
          return !v || validator.isMobilePhone(v);
        },
        message: 'Invalid phone number'
      }
    },
    avatar: String,
    bio: {
      type: String,
      maxlength: 500
    },
    location: {
      country: String,
      state: String,
      city: String,
      timezone: String
    }
  },
  role: {
    type: String,
    enum: ['user', 'premium', 'professional', 'enterprise', 'admin'],
    default: 'user'
  },
  permissions: [{
    type: String,
    enum: ['read', 'write', 'delete', 'admin', 'analytics', 'predictions', 'real_time']
  }],
  preferences: {
    theme: {
      type: String,
      enum: ['light', 'dark', 'auto'],
      default: 'light'
    },
    language: {
      type: String,
      default: 'en'
    },
    currency: {
      type: String,
      default: 'USD'
    },
    timezone: {
      type: String,
      default: 'UTC'
    },
    dateFormat: {
      type: String,
      enum: ['MM/DD/YYYY', 'DD/MM/YYYY', 'YYYY-MM-DD'],
      default: 'MM/DD/YYYY'
    },
    notifications: {
      email: {
        enabled: { type: Boolean, default: true },
        frequency: {
          type: String,
          enum: ['real_time', 'hourly', 'daily', 'weekly'],
          default: 'daily'
        },
        types: [{
          type: String,
          enum: ['alerts', 'predictions', 'news', 'portfolio', 'system']
        }]
      },
      push: {
        enabled: { type: Boolean, default: true },
        types: [{
          type: String,
          enum: ['alerts', 'breaking_news', 'price_targets']
        }]
      },
      sms: {
        enabled: { type: Boolean, default: false },
        types: [{
          type: String,
          enum: ['critical_alerts', 'stop_loss']
        }]
      }
    },
    dashboard: {
      layout: {
        type: String,
        enum: ['compact', 'comfortable', 'spacious'],
        default: 'comfortable'
      },
      widgets: [{
        type: String,
        position: Number,
        size: String,
        settings: mongoose.Schema.Types.Mixed
      }],
      defaultView: {
        type: String,
        enum: ['portfolio', 'watchlist', 'predictions', 'market'],
        default: 'portfolio'
      }
    }
  },
  trading: {
    experience: {
      type: String,
      enum: ['beginner', 'intermediate', 'advanced', 'expert'],
      default: 'beginner'
    },
    riskTolerance: {
      type: String,
      enum: ['conservative', 'moderate', 'aggressive', 'speculative'],
      default: 'moderate'
    },
    investmentGoals: [{
      type: String,
      enum: ['growth', 'income', 'preservation', 'speculation']
    }],
    timeHorizon: {
      type: String,
      enum: ['short_term', 'medium_term', 'long_term'],
      default: 'medium_term'
    },
    autoTrading: {
      enabled: { type: Boolean, default: false },
      strategies: [String],
      maxPositionSize: Number,
      stopLossPercent: Number,
      takeProfitPercent: Number,
      maxDailyTrades: Number
    }
  }
}, {
  timestamps: true,
  toJSON: {
    transform: function(doc, ret) {
      delete ret.password;
      return ret;
    }
  }
});

// Pre-save hook for password hashing
UserSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  
  try {
    const salt = await bcrypt.genSalt(12);
    this.password = await bcrypt.hash(this.password, salt);
    next();
  } catch (error) {
    next(error);
  }
});

// Password comparison method
UserSchema.methods.comparePassword = async function(candidatePassword) {
  return bcrypt.compare(candidatePassword, this.password);
};

// Create models
const Stock = mongoose.model('Stock', StockSchema);
const Prediction = mongoose.model('Prediction', PredictionSchema);
const User = mongoose.model('User', UserSchema);

// Advanced authentication middleware
const authenticateToken = async (req, res, next) => {
  try {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
      return res.status(401).json({
        error: 'Authentication required',
        code: 'AUTH_TOKEN_MISSING'
      });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'fallback_secret');
    
    // Check if token is blacklisted
    const isBlacklisted = await redisClient.get(`blacklist_${token}`);
    if (isBlacklisted) {
      return res.status(401).json({
        error: 'Token has been revoked',
        code: 'AUTH_TOKEN_REVOKED'
      });
    }

    const user = await User.findById(decoded.userId).select('-password');
    if (!user) {
      return res.status(401).json({
        error: 'User not found',
        code: 'AUTH_USER_NOT_FOUND'
      });
    }

    // Check if user is active
    if (user.status === 'suspended' || user.status === 'banned') {
      return res.status(403).json({
        error: 'Account suspended',
        code: 'ACCOUNT_SUSPENDED'
      });
    }

    req.user = user;
    req.token = token;
    next();
  } catch (error) {
    if (error.name === 'TokenExpiredError') {
      return res.status(401).json({
        error: 'Token expired',
        code: 'AUTH_TOKEN_EXPIRED'
      });
    } else if (error.name === 'JsonWebTokenError') {
      return res.status(403).json({
        error: 'Invalid token',
        code: 'AUTH_TOKEN_INVALID'
      });
    } else {
      console.error('Authentication error:', error);
      return res.status(500).json({
        error: 'Internal server error',
        code: 'INTERNAL_ERROR'
      });
    }
  }
};

// Role-based authorization middleware
const requireRole = (roles) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        error: 'Authentication required',
        code: 'AUTH_REQUIRED'
      });
    }

    const userRoles = Array.isArray(req.user.role) ? req.user.role : [req.user.role];
    const requiredRoles = Array.isArray(roles) ? roles : [roles];
    
    const hasRole = requiredRoles.some(role => userRoles.includes(role));
    
    if (!hasRole) {
      return res.status(403).json({
        error: 'Insufficient permissions',
        code: 'INSUFFICIENT_PERMISSIONS',
        required: requiredRoles,
        current: userRoles
      });
    }

    next();
  };
};

// WebSocket connection handling with enhanced features
io.on('connection', (socket) => {
  console.log(`Client connected: ${socket.id}`);

  // Handle authentication for WebSocket
  socket.on('authenticate', async (token) => {
    try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET || 'fallback_secret');
      const user = await User.findById(decoded.userId);
      
      if (user) {
        socket.userId = user._id.toString();
        socket.userRole = user.role;
        socket.emit('authenticated', { success: true, userId: user._id });
        
        // Join user-specific room
        socket.join(`user_${user._id}`);
        
        // Join role-based rooms
        socket.join(`role_${user.role}`);
        
        console.log(`User ${user.email} authenticated on socket ${socket.id}`);
      } else {
        socket.emit('authentication_failed', { error: 'User not found' });
      }
    } catch (error) {
      socket.emit('authentication_failed', { error: 'Invalid token' });
    }
  });

  // Handle room joining for specific stocks
  socket.on('join_stock', (symbol) => {
    if (symbol && typeof symbol === 'string') {
      socket.join(`stock_${symbol.toUpperCase()}`);
      socket.emit('joined_stock', { symbol: symbol.toUpperCase() });
      console.log(`Socket ${socket.id} joined stock room: ${symbol}`);
    }
  });

  socket.on('leave_stock', (symbol) => {
    if (symbol && typeof symbol === 'string') {
      socket.leave(`stock_${symbol.toUpperCase()}`);
      socket.emit('left_stock', { symbol: symbol.toUpperCase() });
      console.log(`Socket ${socket.id} left stock room: ${symbol}`);
    }
  });

  // Handle real-time data subscriptions
  socket.on('subscribe_to_updates', (preferences) => {
    socket.updatePreferences = preferences;
    console.log(`Socket ${socket.id} subscribed to updates:`, preferences);
  });

  // Handle prediction requests
  socket.on('request_prediction', async (data) => {
    if (!socket.userId) {
      socket.emit('prediction_error', { error: 'Authentication required' });
      return;
    }

    try {
      const { symbol, timeframe } = data;
      // Process prediction request (implementation would call ML service)
      socket.emit('prediction_processing', { symbol, timeframe, status: 'processing' });
    } catch (error) {
      socket.emit('prediction_error', { error: error.message });
    }
  });

  socket.on('disconnect', (reason) => {
    console.log(`Client disconnected: ${socket.id}, reason: ${reason}`);
    
    // Clean up any pending operations
    if (socket.userId) {
      // Update user's last seen timestamp
      User.findByIdAndUpdate(socket.userId, {
        lastSeen: new Date(),
        isOnline: false
      }).catch(err => console.error('Error updating user last seen:', err));
    }
  });

  // Error handling
  socket.on('error', (error) => {
    console.error(`Socket error for ${socket.id}:`, error);
  });
});

// Broadcasting utilities
const broadcastToStock = (symbol, event, data) => {
  io.to(`stock_${symbol.toUpperCase()}`).emit(event, data);
};

const broadcastToUser = (userId, event, data) => {
  io.to(`user_${userId}`).emit(event, data);
};

const broadcastToRole = (role, event, data) => {
  io.to(`role_${role}`).emit(event, data);
};

const broadcastGlobal = (event, data) => {
  io.emit(event, data);
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: process.env.npm_package_version || '1.0.0',
    environment: process.env.NODE_ENV || 'development'
  });
});

// API health check
app.get('/api/health', async (req, res) => {
  const health = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    services: {}
  };

  try {
    // Check MongoDB
    const mongoState = mongoose.connection.readyState;
    health.services.mongodb = {
      status: mongoState === 1 ? 'connected' : 'disconnected',
      state: mongoState
    };

    // Check Redis
    try {
      await redisClient.ping();
      health.services.redis = { status: 'connected' };
    } catch (redisError) {
      health.services.redis = { status: 'disconnected', error: redisError.message };
      health.status = 'degraded';
    }

    // Check external APIs
    health.services.externalAPIs = {
      alphaVantage: !!process.env.ALPHA_VANTAGE_API_KEY,
      polygonIO: !!process.env.POLYGON_API_KEY
    };

    res.status(health.status === 'healthy' ? 200 : 503).json(health);
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Start server
const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`WebSocket server enabled`);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down gracefully');
  
  server.close(() => {
    console.log('HTTP server closed');
  });
  
  await mongoose.connection.close();
  await redisClient.quit();
  process.exit(0);
});

export default app;