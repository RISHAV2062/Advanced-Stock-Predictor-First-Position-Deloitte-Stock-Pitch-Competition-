import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  DollarSign, 
  BarChart3, 
  PieChart, 
  Bell, 
  Settings, 
  Search, 
  Filter, 
  RefreshCw, 
  Star, 
  Eye, 
  AlertTriangle, 
  Target, 
  Zap,
  Calendar,
  Clock,
  Globe,
  Users,
  Shield,
  Database,
  Wifi,
  WifiOff,
  CheckCircle,
  XCircle,
  Minus,
  Plus,
  ArrowUp,
  ArrowDown,
  ArrowRight,
  Info,
  HelpCircle,
  Download,
  Upload,
  Share2,
  Bookmark,
  Heart,
  MessageSquare,
  Phone,
  Mail,
  Lock,
  Unlock,
  Key,
  User,
  CreditCard,
  Package,
  Truck,
  MapPin,
  Camera,
  Image,
  Video,
  Music,
  File,
  FileText,
  Folder,
  Archive,
  Trash2,
  Edit3,
  Copy,
  Scissors,
  Clipboard,
  Link,
  ExternalLink,
  Home,
  Building,
  Car,
  Plane,
  Train,
  Bus,
  Bike,
  Coffee,
  Gift,
  Gamepad2,
  Headphones,
  Laptop,
  Monitor,
  Printer,
  Router,
  Server,
  Smartphone,
  Tablet,
  Watch,
  Battery,
  Power,
  Plug,
  Bluetooth,
  Rss,
  Bookmark as BookmarkIcon,
  Flag,
  Tag,
  Hash,
  AtSign,
  Percent,
  Euro,
  Pound,
  Yen,
  Bitcoin,
  Coins,
  Wallet,
  CreditCard as CreditCardIcon,
  Receipt,
  ShoppingCart,
  ShoppingBag,
  Store,
  Award,
  Trophy,
  Medal,
  Crown,
  Gem,
  Sparkles,
  Sun,
  Moon,
  CloudRain,
  Cloud,
  Snowflake,
  Wind,
  Thermometer,
  Droplet,
  Umbrella,
  Sunrise,
  Sunset,
  Navigation,
  Anchor,
  Compass,
  Map,
  MapPin2,
  Route,
  Footprints,
  Mountain,
  Trees,
  Flower,
  Leaf,
  Sprout,
  Bug,
  Fish,
  Bird,
  Paw,
  Bone,
  Feather
} from 'lucide-react';
import io from 'socket.io-client';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler
} from 'chart.js';
import { Line, Bar, Pie, Doughnut, Scatter } from 'react-chartjs-2';
import 'chartjs-adapter-moment';
import moment from 'moment';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler
);

// Constants and configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';
const WEBSOCKET_URL = process.env.REACT_APP_WS_URL || 'http://localhost:3001';

const CHART_COLORS = {
  primary: '#3B82F6',
  secondary: '#10B981',
  danger: '#EF4444',
  warning: '#F59E0B',
  info: '#06B6D4',
  purple: '#8B5CF6',
  pink: '#EC4899',
  indigo: '#6366F1',
  gray: '#6B7280'
};

const TIMEFRAMES = [
  { value: '1H', label: '1 Hour', duration: 60 * 60 * 1000 },
  { value: '4H', label: '4 Hours', duration: 4 * 60 * 60 * 1000 },
  { value: '1D', label: '1 Day', duration: 24 * 60 * 60 * 1000 },
  { value: '1W', label: '1 Week', duration: 7 * 24 * 60 * 60 * 1000 },
  { value: '1M', label: '1 Month', duration: 30 * 24 * 60 * 60 * 1000 },
  { value: '3M', label: '3 Months', duration: 90 * 24 * 60 * 60 * 1000 },
  { value: '6M', label: '6 Months', duration: 180 * 24 * 60 * 60 * 1000 },
  { value: '1Y', label: '1 Year', duration: 365 * 24 * 60 * 60 * 1000 }
];

const RISK_LEVELS = {
  very_low: { color: '#10B981', label: 'Very Low', icon: Shield },
  low: { color: '#34D399', label: 'Low', icon: CheckCircle },
  moderate: { color: '#FBBF24', label: 'Moderate', icon: Minus },
  high: { color: '#F87171', label: 'High', icon: AlertTriangle },
  very_high: { color: '#EF4444', label: 'Very High', icon: XCircle }
};

// Custom hooks
const useWebSocket = (url, token) => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!token) return;

    const newSocket = io(url, {
      transports: ['websocket', 'polling'],
      timeout: 20000,
      forceNew: true
    });

    newSocket.on('connect', () => {
      console.log('WebSocket connected');
      setConnected(true);
      setError(null);
      
      // Authenticate with the server
      newSocket.emit('authenticate', token);
    });

    newSocket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      setConnected(false);
      
      if (reason === 'io server disconnect') {
        // Server disconnected, try to reconnect
        newSocket.connect();
      }
    });

    newSocket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      setError(error.message);
      setConnected(false);
    });

    newSocket.on('authenticated', (data) => {
      console.log('WebSocket authenticated:', data);
    });

    newSocket.on('authentication_failed', (data) => {
      console.error('WebSocket authentication failed:', data);
      setError(data.error);
    });

    setSocket(newSocket);

    return () => {
      if (newSocket) {
        newSocket.disconnect();
      }
    };
  }, [url, token]);

  return { socket, connected, error };
};

const useLocalStorage = (key, initialValue) => {
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  const setValue = useCallback((value) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  }, [key, storedValue]);

  return [storedValue, setValue];
};

const useDebounce = (value, delay) => {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

const useApi = () => {
  const [token, setToken] = useLocalStorage('auth_token', null);
  
  const api = useMemo(() => {
    const instance = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // Request interceptor to add auth token
    instance.interceptors.request.use(
      (config) => {
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    instance.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          setToken(null);
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );

    return instance;
  }, [token, setToken]);

  return { api, token, setToken };
};

// Utility functions
const formatCurrency = (value, currency = 'USD') => {
  if (value === null || value === undefined) return 'N/A';
  
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(value);
};

const formatPercentage = (value, decimals = 2) => {
  if (value === null || value === undefined) return 'N/A';
  
  return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`;
};

const formatNumber = (value, decimals = 0) => {
  if (value === null || value === undefined) return 'N/A';
  
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  }).format(value);
};

const formatMarketCap = (value) => {
  if (!value) return 'N/A';
  
  if (value >= 1e12) {
    return `$${(value / 1e12).toFixed(2)}T`;
  } else if (value >= 1e9) {
    return `$${(value / 1e9).toFixed(2)}B`;
  } else if (value >= 1e6) {
    return `$${(value / 1e6).toFixed(2)}M`;
  } else if (value >= 1e3) {
    return `$${(value / 1e3).toFixed(2)}K`;
  } else {
    return formatCurrency(value);
  }
};

const formatTimeAgo = (date) => {
  return moment(date).fromNow();
};

const calculateReturn = (currentPrice, initialPrice) => {
  if (!currentPrice || !initialPrice) return 0;
  return ((currentPrice - initialPrice) / initialPrice) * 100;
};

const getColorForChange = (value) => {
  if (value > 0) return CHART_COLORS.secondary;
  if (value < 0) return CHART_COLORS.danger;
  return CHART_COLORS.gray;
};

const getRiskColor = (riskLevel) => {
  return RISK_LEVELS[riskLevel]?.color || CHART_COLORS.gray;
};

// Components
const LoadingSpinner = ({ size = 'md', className = '' }) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  };

  return (
    <div className={`flex items-center justify-center ${className}`}>
      <div className={`${sizeClasses[size]} animate-spin rounded-full border-2 border-gray-300 border-t-blue-600`}></div>
    </div>
  );
};

const Card = ({ children, className = '', padding = 'p-6', hover = true }) => {
  return (
    <div className={`
      bg-white rounded-xl shadow-sm border border-gray-200 
      ${hover ? 'hover:shadow-md transition-shadow duration-200' : ''}
      ${padding} ${className}
    `}>
      {children}
    </div>
  );
};

const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  disabled = false, 
  loading = false,
  icon: Icon,
  onClick,
  className = '',
  ...props 
}) => {
  const baseClasses = 'inline-flex items-center justify-center font-medium rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2';
  
  const variants = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500 disabled:bg-blue-300',
    secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200 focus:ring-gray-500 disabled:bg-gray-50',
    success: 'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500 disabled:bg-green-300',
    danger: 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500 disabled:bg-red-300',
    warning: 'bg-yellow-600 text-white hover:bg-yellow-700 focus:ring-yellow-500 disabled:bg-yellow-300',
    outline: 'border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 focus:ring-blue-500 disabled:bg-gray-50'
  };
  
  const sizes = {
    xs: 'px-2.5 py-1.5 text-xs',
    sm: 'px-3 py-2 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-4 py-2 text-base',
    xl: 'px-6 py-3 text-base'
  };

  return (
    <button
      className={`${baseClasses} ${variants[variant]} ${sizes[size]} ${disabled || loading ? 'cursor-not-allowed' : ''} ${className}`}
      disabled={disabled || loading}
      onClick={onClick}
      {...props}
    >
      {loading && <RefreshCw className="w-4 h-4 mr-2 animate-spin" />}
      {!loading && Icon && <Icon className="w-4 h-4 mr-2" />}
      {children}
    </button>
  );
};

const Badge = ({ children, variant = 'default', size = 'sm', className = '' }) => {
  const variants = {
    default: 'bg-gray-100 text-gray-800',
    primary: 'bg-blue-100 text-blue-800',
    secondary: 'bg-gray-100 text-gray-800',
    success: 'bg-green-100 text-green-800',
    danger: 'bg-red-100 text-red-800',
    warning: 'bg-yellow-100 text-yellow-800',
    info: 'bg-cyan-100 text-cyan-800'
  };
  
  const sizes = {
    xs: 'px-2 py-0.5 text-xs',
    sm: 'px-2.5 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-3.5 py-1.5 text-sm'
  };

  return (
    <span className={`inline-flex items-center font-medium rounded-full ${variants[variant]} ${sizes[size]} ${className}`}>
      {children}
    </span>
  );
};

const ProgressBar = ({ value, max = 100, className = '', showLabel = true, color = 'blue' }) => {
  const percentage = Math.min((value / max) * 100, 100);
  
  const colors = {
    blue: 'bg-blue-600',
    green: 'bg-green-600',
    red: 'bg-red-600',
    yellow: 'bg-yellow-600',
    purple: 'bg-purple-600'
  };

  return (
    <div className={`w-full ${className}`}>
      {showLabel && (
        <div className="flex justify-between mb-1">
          <span className="text-sm font-medium text-gray-700">{percentage.toFixed(1)}%</span>
        </div>
      )}
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div 
          className={`h-2 rounded-full transition-all duration-300 ${colors[color]}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

const Tooltip = ({ children, content, position = 'top' }) => {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
      >
        {children}
      </div>
      {isVisible && (
        <div className={`
          absolute z-10 px-3 py-2 text-sm text-white bg-gray-900 rounded-lg shadow-lg
          ${position === 'top' ? 'bottom-full mb-2 left-1/2 transform -translate-x-1/2' : ''}
          ${position === 'bottom' ? 'top-full mt-2 left-1/2 transform -translate-x-1/2' : ''}
          ${position === 'left' ? 'right-full mr-2 top-1/2 transform -translate-y-1/2' : ''}
          ${position === 'right' ? 'left-full ml-2 top-1/2 transform -translate-y-1/2' : ''}
        `}>
          {content}
          <div className={`
            absolute w-2 h-2 bg-gray-900 transform rotate-45
            ${position === 'top' ? 'top-full left-1/2 transform -translate-x-1/2 -translate-y-1/2' : ''}
            ${position === 'bottom' ? 'bottom-full left-1/2 transform -translate-x-1/2 translate-y-1/2' : ''}
            ${position === 'left' ? 'left-full top-1/2 transform -translate-x-1/2 -translate-y-1/2' : ''}
            ${position === 'right' ? 'right-full top-1/2 transform translate-x-1/2 -translate-y-1/2' : ''}
          `} />
        </div>
      )}
    </div>
  );
};

const Modal = ({ isOpen, onClose, title, children, size = 'md' }) => {
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }

    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const sizes = {
    sm: 'max-w-md',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
    xl: 'max-w-4xl',
    full: 'max-w-full mx-4'
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div 
        className="absolute inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={onClose}
      />
      <div className={`relative bg-white rounded-xl shadow-xl w-full ${sizes[size]} max-h-[90vh] overflow-hidden`}>
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
          >
            <XCircle className="w-5 h-5" />
          </button>
        </div>
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
          {children}
        </div>
      </div>
    </div>
  );
};

const SearchInput = ({ 
  value, 
  onChange, 
  placeholder = "Search...", 
  className = "",
  onFocus,
  onBlur,
  loading = false 
}) => {
  return (
    <div className={`relative ${className}`}>
      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
        {loading ? (
          <RefreshCw className="w-5 h-5 text-gray-400 animate-spin" />
        ) : (
          <Search className="w-5 h-5 text-gray-400" />
        )}
      </div>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={onFocus}
        onBlur={onBlur}
        placeholder={placeholder}
        className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      />
    </div>
  );
};

const StockCard = ({ stock, onClick, showActions = true }) => {
  const changeColor = getColorForChange(stock.changePercent);
  const ChangeIcon = stock.changePercent > 0 ? TrendingUp : stock.changePercent < 0 ? TrendingDown : Minus;

  return (
    <Card 
      className="cursor-pointer hover:shadow-lg transition-all duration-200" 
      onClick={() => onClick(stock)}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-3">
            <div>
              <h3 className="font-semibold text-gray-900">{stock.symbol}</h3>
              <p className="text-sm text-gray-600 truncate">{stock.name}</p>
            </div>
          </div>
          <div className="mt-3 flex items-center justify-between">
            <div>
              <p className="text-2xl font-bold text-gray-900">
                {formatCurrency(stock.price)}
              </p>
              <div className="flex items-center space-x-1 mt-1">
                <ChangeIcon className="w-4 h-4" style={{ color: changeColor }} />
                <span className="text-sm font-medium" style={{ color: changeColor }}>
                  {formatCurrency(stock.change)} ({formatPercentage(stock.changePercent)})
                </span>
              </div>
            </div>
            {showActions && (
              <div className="flex items-center space-x-2">
                <Tooltip content="Add to watchlist">
                  <button className="p-2 text-gray-400 hover:text-yellow-500 transition-colors">
                    <Star className="w-5 h-5" />
                  </button>
                </Tooltip>
                <Tooltip content="View details">
                  <button className="p-2 text-gray-400 hover:text-blue-500 transition-colors">
                    <Eye className="w-5 h-5" />
                  </button>
                </Tooltip>
              </div>
            )}
          </div>
        </div>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-gray-500">Volume:</span>
          <span className="ml-2 font-medium">{formatNumber(stock.volume)}</span>
        </div>
        <div>
          <span className="text-gray-500">Market Cap:</span>
          <span className="ml-2 font-medium">{formatMarketCap(stock.marketCap)}</span>
        </div>
      </div>
    </Card>
  );
};

const PredictionCard = ({ prediction, className = "" }) => {
  const directionColor = prediction.direction === 'bullish' ? CHART_COLORS.secondary : 
                        prediction.direction === 'bearish' ? CHART_COLORS.danger : 
                        CHART_COLORS.gray;
  
  const DirectionIcon = prediction.direction === 'bullish' ? TrendingUp : 
                       prediction.direction === 'bearish' ? TrendingDown : 
                       Minus;

  return (
    <Card className={`${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="font-semibold text-gray-900">{prediction.symbol}</h3>
          <p className="text-sm text-gray-600">{prediction.timeframe} Prediction</p>
        </div>
        <Badge variant={prediction.direction === 'bullish' ? 'success' : prediction.direction === 'bearish' ? 'danger' : 'secondary'}>
          <DirectionIcon className="w-3 h-3 mr-1" />
          {prediction.direction}
        </Badge>
      </div>

      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">Current Price:</span>
          <span className="font-semibold">{formatCurrency(prediction.currentPrice)}</span>
        </div>
        
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">Predicted Price:</span>
          <span className="font-semibold" style={{ color: directionColor }}>
            {formatCurrency(prediction.predictedPrice)}
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">Expected Return:</span>
          <span className="font-semibold" style={{ color: directionColor }}>
            {formatPercentage(((prediction.predictedPrice - prediction.currentPrice) / prediction.currentPrice) * 100)}
          </span>
        </div>

        <div className="mt-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-gray-600">Confidence:</span>
            <span className="text-sm font-medium">{prediction.confidence}%</span>
          </div>
          <ProgressBar 
            value={prediction.confidence} 
            color={prediction.confidence > 70 ? 'green' : prediction.confidence > 40 ? 'yellow' : 'red'}
            showLabel={false}
          />
        </div>

        <div className="mt-4 text-xs text-gray-500">
          <div className="flex justify-between">
            <span>Model: {prediction.modelPerformance?.algorithm}</span>
            <span>Created: {formatTimeAgo(prediction.createdAt)}</span>
          </div>
        </div>
      </div>
    </Card>
  );
};

const TechnicalIndicatorCard = ({ indicator, title, value, signal, description }) => {
  const getSignalColor = (signal) => {
    switch (signal?.toLowerCase()) {
      case 'buy':
      case 'bullish':
      case 'oversold':
        return CHART_COLORS.secondary;
      case 'sell':
      case 'bearish':
      case 'overbought':
        return CHART_COLORS.danger;
      default:
        return CHART_COLORS.gray;
    }
  };

  return (
    <Card padding="p-4">
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-medium text-gray-900">{title}</h4>
        {signal && (
          <Badge 
            variant={signal.toLowerCase().includes('buy') || signal.toLowerCase().includes('bullish') ? 'success' : 
                    signal.toLowerCase().includes('sell') || signal.toLowerCase().includes('bearish') ? 'danger' : 'secondary'}
          >
            {signal}
          </Badge>
        )}
      </div>
      <div className="space-y-1">
        <p className="text-lg font-semibold" style={{ color: getSignalColor(signal) }}>
          {typeof value === 'number' ? value.toFixed(2) : value}
        </p>
        {description && (
          <p className="text-xs text-gray-600">{description}</p>
        )}
      </div>
    </Card>
  );
};

// Advanced Chart Components
const PriceChart = ({ data, symbol, timeframe = '1D', height = 400 }) => {
  const chartData = useMemo(() => {
    if (!data || !data.length) return null;

    return {
      labels: data.map(d => moment(d.timestamp).format(timeframe === '1D' ? 'HH:mm' : 'MM/DD')),
      datasets: [
        {
          label: 'Price',
          data: data.map(d => d.close),
          borderColor: CHART_COLORS.primary,
          backgroundColor: `${CHART_COLORS.primary}20`,
          borderWidth: 2,
          fill: true,
          tension: 0.1,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointBackgroundColor: CHART_COLORS.primary,
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2
        }
      ]
    };
  }, [data, timeframe]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: CHART_COLORS.primary,
        borderWidth: 1
      }
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: false
        },
        ticks: {
          maxTicksLimit: 10
        }
      },
      y: {
        display: true,
        position: 'right',
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          callback: function(value) {
            return formatCurrency(value);
          }
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  if (!chartData) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div style={{ height }}>
      <Line data={chartData} options={options} />
    </div>
  );
};

const VolumeChart = ({ data, height = 200 }) => {
  const chartData = useMemo(() => {
    if (!data || !data.length) return null;

    return {
      labels: data.map(d => moment(d.timestamp).format('MM/DD')),
      datasets: [
        {
          label: 'Volume',
          data: data.map(d => d.volume),
          backgroundColor: data.map(d => d.close > d.open ? CHART_COLORS.secondary : CHART_COLORS.danger),
          borderWidth: 0
        }
      ]
    };
  }, [data]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      }
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: false
        }
      },
      y: {
        display: true,
        position: 'right',
        ticks: {
          callback: function(value) {
            return formatNumber(value / 1000000, 1) + 'M';
          }
        }
      }
    }
  };

  if (!chartData) {
    return <LoadingSpinner size="lg" />;
  }

  return (
    <div style={{ height }}>
      <Bar data={chartData} options={options} />
    </div>
  );
};

// Main App Component
function App() {
  const { api, token, setToken } = useApi();
  const { socket, connected } = useWebSocket(WEBSOCKET_URL, token);
  
  // State management
  const [user, setUser] = useState(null);
  const [currentView, setCurrentView] = useState('dashboard');
  const [selectedStock, setSelectedStock] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Market data state
  const [marketData, setMarketData] = useState({});
  const [predictions, setPredictions] = useState([]);
  const [watchlist, setWatchlist] = useState([]);
  const [portfolio, setPortfolio] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [news, setNews] = useState([]);
  
  // UI state
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D');
  const [showPredictionModal, setShowPredictionModal] = useState(false);
  const [showAlertModal, setShowAlertModal] = useState(false);
  
  // Debounced search
  const debouncedSearchQuery = useDebounce(searchQuery, 300);

  // Initialize app
  useEffect(() => {
    if (token) {
      initializeApp();
    }
  }, [token]);

  // Handle WebSocket events
  useEffect(() => {
    if (!socket) return;

    socket.on('market_update', handleMarketUpdate);
    socket.on('prediction_update', handlePredictionUpdate);
    socket.on('alert', handleAlert);
    socket.on('news_update', handleNewsUpdate);

    return () => {
      socket.off('market_update');
      socket.off('prediction_update');
      socket.off('alert');
      socket.off('news_update');
    };
  }, [socket]);

  // Search functionality
  useEffect(() => {
    if (debouncedSearchQuery.length > 2) {
      searchStocks(debouncedSearchQuery);
    } else {
      setSearchResults([]);
    }
  }, [debouncedSearchQuery]);

  // API functions
  const initializeApp = async () => {
    try {
      setLoading(true);
      
      // Fetch user data
      const userResponse = await api.get('/user/profile');
      setUser(userResponse.data);
      
      // Fetch initial data
      await Promise.all([
        fetchWatchlist(),
        fetchPortfolio(),
        fetchAlerts(),
        fetchRecentPredictions(),
        fetchMarketNews()
      ]);
      
    } catch (error) {
      console.error('Error initializing app:', error);
      setError('Failed to load application data');
    } finally {
      setLoading(false);
    }
  };

  const searchStocks = async (query) => {
    try {
      const response = await api.get(`/stocks/search?q=${encodeURIComponent(query)}`);
      setSearchResults(response.data.results || []);
    } catch (error) {
      console.error('Error searching stocks:', error);
    }
  };

  const fetchWatchlist = async () => {
    try {
      const response = await api.get('/user/watchlist');
      setWatchlist(response.data.watchlist || []);
    } catch (error) {
      console.error('Error fetching watchlist:', error);
    }
  };

  const fetchPortfolio = async () => {
    try {
      const response = await api.get('/portfolio');
      setPortfolio(response.data.portfolio || []);
    } catch (error) {
      console.error('Error fetching portfolio:', error);
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await api.get('/alerts');
      setAlerts(response.data.alerts || []);
    } catch (error) {
      console.error('Error fetching alerts:', error);
    }
  };

  const fetchRecentPredictions = async () => {
    try {
      const response = await api.get('/predictions?limit=10');
      setPredictions(response.data.predictions || []);
    } catch (error) {
      console.error('Error fetching predictions:', error);
    }
  };

  const fetchMarketNews = async () => {
    try {
      const response = await api.get('/news?limit=20');
      setNews(response.data.news || []);
    } catch (error) {
      console.error('Error fetching news:', error);
    }
  };

  const fetchStockData = async (symbol) => {
    try {
      setLoading(true);
      const response = await api.get(`/stocks/${symbol}`);
      setSelectedStock(response.data);
      
      // Fetch predictions for this stock
      const predictionResponse = await api.get(`/predictions/${symbol}`);
      // Handle prediction data...
      
    } catch (error) {
      console.error('Error fetching stock data:', error);
      setError(`Failed to load data for ${symbol}`);
    } finally {
      setLoading(false);
    }
  };

  // WebSocket event handlers
  const handleMarketUpdate = (data) => {
    setMarketData(prev => ({
      ...prev,
      [data.symbol]: data
    }));
  };

  const handlePredictionUpdate = (prediction) => {
    setPredictions(prev => [prediction, ...prev.slice(0, 9)]);
  };

  const handleAlert = (alert) => {
    setAlerts(prev => [alert, ...prev]);
    
    // Show notification
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(`Stock Alert: ${alert.symbol}`, {
        body: alert.message,
        icon: '/favicon.ico'
      });
    }
  };

  const handleNewsUpdate = (newsItem) => {
    setNews(prev => [newsItem, ...prev.slice(0, 19)]);
  };

  // UI Components
  const Sidebar = () => (
    <div className={`fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-200 ease-in-out ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
      <div className="flex items-center justify-between p-6 border-b border-gray-200">
        <h1 className="text-xl font-bold text-gray-900">StockPredictor</h1>
        <button
          onClick={() => setSidebarOpen(false)}
          className="p-2 text-gray-400 hover:text-gray-600 lg:hidden"
        >
          <XCircle className="w-5 h-5" />
        </button>
      </div>
      
      <nav className="mt-6">
        <div className="px-6 space-y-1">
          {[
            { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
            { id: 'watchlist', label: 'Watchlist', icon: Eye },
            { id: 'portfolio', label: 'Portfolio', icon: PieChart },
            { id: 'predictions', label: 'Predictions', icon: Target },
            { id: 'alerts', label: 'Alerts', icon: Bell },
            { id: 'news', label: 'News', icon: Globe },
            { id: 'settings', label: 'Settings', icon: Settings }
          ].map(item => {
            const IconComponent = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => setCurrentView(item.id)}
                className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                  currentView === item.id 
                    ? 'bg-blue-100 text-blue-700' 
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }`}
              >
                <IconComponent className="w-5 h-5 mr-3" />
                {item.label}
              </button>
            );
          })}
        </div>
      </nav>
      
      <div className="absolute bottom-0 left-0 right-0 p-6 border-t border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
            <User className="w-4 h-4 text-white" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900 truncate">
              {user?.profile?.firstName || user?.email}
            </p>
            <p className="text-xs text-gray-500 truncate">{user?.role}</p>
          </div>
        </div>
      </div>
    </div>
  );

  const Header = () => (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="flex items-center justify-between px-6 py-4">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 text-gray-400 hover:text-gray-600 lg:hidden"
          >
            <BarChart3 className="w-5 h-5" />
          </button>
          
          <SearchInput
            value={searchQuery}
            onChange={setSearchQuery}
            placeholder="Search stocks..."
            className="w-96"
          />
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            {connected ? (
              <Tooltip content="Connected to real-time data">
                <Wifi className="w-5 h-5 text-green-500" />
              </Tooltip>
            ) : (
              <Tooltip content="Disconnected from real-time data">
                <WifiOff className="w-5 h-5 text-red-500" />
              </Tooltip>
            )}
          </div>
          
          <Button
            variant="outline"
            size="sm"
            icon={Bell}
            onClick={() => setShowAlertModal(true)}
          >
            Alerts {alerts.length > 0 && `(${alerts.length})`}
          </Button>
          
          <Button
            variant="primary"
            size="sm"
            icon={Target}
            onClick={() => setShowPredictionModal(true)}
          >
            New Prediction
          </Button>
        </div>
      </div>
      
      {searchResults.length > 0 && (
        <div className="absolute top-full left-0 right-0 bg-white shadow-lg border-t border-gray-200 z-40">
          <div className="max-h-96 overflow-y-auto">
            {searchResults.map(stock => (
              <div
                key={stock.symbol}
                onClick={() => {
                  fetchStockData(stock.symbol);
                  setSearchQuery('');
                  setSearchResults([]);
                }}
                className="flex items-center justify-between p-4 hover:bg-gray-50 cursor-pointer border-b border-gray-100"
              >
                <div>
                  <p className="font-medium text-gray-900">{stock.symbol}</p>
                  <p className="text-sm text-gray-600">{stock.name}</p>
                </div>
                <div className="text-right">
                  <p className="font-medium">{formatCurrency(stock.price)}</p>
                  <p className={`text-sm ${stock.changePercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatPercentage(stock.changePercent)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </header>
  );

  const Dashboard = () => (
    <div className="space-y-6">
      {/* Market Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <div className="flex items-center">
            <div className="p-2 bg-blue-100 rounded-lg">
              <TrendingUp className="w-6 h-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-600">Portfolio Value</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatCurrency(portfolio.reduce((sum, item) => sum + (item.currentValue || 0), 0))}
              </p>
            </div>
          </div>
        </Card>
        
        <Card>
          <div className="flex items-center">
            <div className="p-2 bg-green-100 rounded-lg">
              <DollarSign className="w-6 h-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-600">Total Gain/Loss</p>
              <p className="text-2xl font-bold text-green-600">
                {formatCurrency(12450)}
              </p>
            </div>
          </div>
        </Card>
        
        <Card>
          <div className="flex items-center">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Target className="w-6 h-6 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-600">Active Predictions</p>
              <p className="text-2xl font-bold text-gray-900">{predictions.length}</p>
            </div>
          </div>
        </Card>
        
        <Card>
          <div className="flex items-center">
            <div className="p-2 bg-yellow-100 rounded-lg">
              <Bell className="w-6 h-6 text-yellow-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-600">Active Alerts</p>
              <p className="text-2xl font-bold text-gray-900">{alerts.length}</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Recent Predictions */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Recent Predictions</h2>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCurrentView('predictions')}
          >
            View All
          </Button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {predictions.slice(0, 6).map(prediction => (
            <PredictionCard key={prediction._id} prediction={prediction} />
          ))}
        </div>
      </div>

      {/* Watchlist */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Watchlist</h2>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCurrentView('watchlist')}
          >
            View All
          </Button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {watchlist.slice(0, 6).map(stock => (
            <StockCard 
              key={stock.symbol} 
              stock={stock} 
              onClick={fetchStockData}
            />
          ))}
        </div>
      </div>
    </div>
  );

  const StockDetail = () => {
    if (!selectedStock) return null;

    return (
      <div className="space-y-6">
        {/* Stock Header */}
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">{selectedStock.symbol}</h1>
              <p className="text-gray-600">{selectedStock.name}</p>
            </div>
            <div className="text-right">
              <p className="text-3xl font-bold text-gray-900">
                {formatCurrency(selectedStock.price)}
              </p>
              <div className="flex items-center justify-end space-x-1 mt-1">
                {selectedStock.changePercent > 0 ? (
                  <TrendingUp className="w-5 h-5 text-green-600" />
                ) : (
                  <TrendingDown className="w-5 h-5 text-red-600" />
                )}
                <span className={`text-lg font-medium ${selectedStock.changePercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {formatCurrency(selectedStock.change)} ({formatPercentage(selectedStock.changePercent)})
                </span>
              </div>
            </div>
          </div>
        </Card>

        {/* Chart */}
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Price Chart</h2>
            <div className="flex space-x-1">
              {TIMEFRAMES.map(timeframe => (
                <button
                  key={timeframe.value}
                  onClick={() => setSelectedTimeframe(timeframe.value)}
                  className={`px-3 py-1 text-sm font-medium rounded ${
                    selectedTimeframe === timeframe.value
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  {timeframe.label}
                </button>
              ))}
            </div>
          </div>
          <PriceChart 
            data={selectedStock.historicalData} 
            symbol={selectedStock.symbol}
            timeframe={selectedTimeframe}
          />
        </Card>

        {/* Technical Indicators */}
        <div>
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Technical Analysis</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <TechnicalIndicatorCard
              title="RSI"
              value={selectedStock.technicalIndicators?.rsi?.value}
              signal={selectedStock.technicalIndicators?.rsi?.signal}
              description="Relative Strength Index"
            />
            <TechnicalIndicatorCard
              title="MACD"
              value={selectedStock.technicalIndicators?.macd?.macd}
              signal={selectedStock.technicalIndicators?.macd?.trend}
              description="Moving Average Convergence Divergence"
            />
            <TechnicalIndicatorCard
              title="SMA 20"
              value={selectedStock.technicalIndicators?.movingAverages?.sma20}
              description="20-day Simple Moving Average"
            />
            <TechnicalIndicatorCard
              title="Volume"
              value={formatNumber(selectedStock.volume)}
              description="Trading Volume"
            />
          </div>
        </div>
      </div>
    );
  };

  // Main render
  if (loading && !user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="xl" />
      </div>
    );
  }

  if (!token) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <Card className="w-full max-w-md">
          <div className="text-center">
            <h1 className="text-2xl font-bold text-gray-900 mb-6">Stock Predictor</h1>
            <p className="text-gray-600 mb-6">Please log in to access the platform</p>
            <Button
              variant="primary"
              size="lg"
              className="w-full"
              onClick={() => setToken('demo-token')} // Demo token for testing
            >
              Demo Login
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar />
      
      <div className={`transition-all duration-200 ${sidebarOpen ? 'lg:ml-64' : ''}`}>
        <Header />
        
        <main className="p-6">
          {error && (
            <div className="mb-6 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
              {error}
              <button
                onClick={() => setError(null)}
                className="float-right font-bold text-red-700 hover:text-red-900"
              >
                ×
              </button>
            </div>
          )}
          
          {currentView === 'dashboard' && <Dashboard />}
          {selectedStock && <StockDetail />}
          {/* Add other views here */}
        </main>
      </div>
      
      {/* Modals */}
      <Modal
        isOpen={showPredictionModal}
        onClose={() => setShowPredictionModal(false)}
        title="Create New Prediction"
        size="lg"
      >
        <div className="space-y-4">
          <p>Prediction modal content...</p>
        </div>
      </Modal>
      
      <Modal
        isOpen={showAlertModal}
        onClose={() => setShowAlertModal(false)}
        title="Manage Alerts"
        size="lg"
      >
        <div className="space-y-4">
          <p>Alert management content...</p>
        </div>
      </Modal>
    </div>
  );
}

export default App;