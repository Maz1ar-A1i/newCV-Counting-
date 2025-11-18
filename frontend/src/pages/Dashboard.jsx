import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Grid, 
  useTheme, 
  ToggleButton, 
  ToggleButtonGroup,
  Card,
  CardContent,
  Chip,
  Divider,
  IconButton,
  CircularProgress,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Avatar,
  Badge,
  Tooltip,
  alpha,
  Stack,
  Skeleton
} from '@mui/material';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, 
  ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area,
  RadialBarChart, RadialBar, Legend, Sector, ReferenceLine
} from 'recharts';
import axios from 'axios';
import { format, parseISO, formatDistanceToNow } from 'date-fns';
import { 
  VideoCamera, ChartLine, Calendar, ArrowUp, ArrowDown, 
  Lightning, Timer, Eye, TrendUp, CaretUp, CaretDown,
  Camera as CameraIcon, Pulse, ArrowClockwise
} from '@phosphor-icons/react';
import { styled } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';

const StyledToggleButtonGroup = styled(ToggleButtonGroup)(({ theme }) => ({
  backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.04)',
  borderRadius: '12px',
  padding: '4px',
  '& .MuiToggleButton-root': {
    border: 'none',
    borderRadius: '8px',
    padding: '6px 16px',
    textTransform: 'none',
    fontSize: '0.875rem',
    fontWeight: 500,
    color: theme.palette.text.secondary,
    transition: 'all 0.2s',
    '&:hover': {
      backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.08)',
    },
    '&.Mui-selected': {
      backgroundColor: theme.palette.mode === 'dark' ? theme.palette.grey[800] : theme.palette.common.white,
      color: theme.palette.text.primary,
      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
      '&:hover': {
        backgroundColor: theme.palette.mode === 'dark' ? theme.palette.grey[800] : theme.palette.common.white,
      },
    },
  },
}));

const StatCard = styled(Card)(({ theme }) => ({
  height: '160px',
  background: 'transparent',
  backdropFilter: 'blur(10px)',
  border: 'none',
  boxShadow: 'none',
  borderRadius: theme.shape.borderRadius * 3,
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  '&:hover': {
    transform: 'translateY(-4px)',
  }
}));

const MetricValue = styled(Typography)(({ theme }) => ({
  fontSize: '2rem',
  fontWeight: 700,
  lineHeight: 1.2,
  color: theme.palette.text.primary,
}));

const MetricLabel = styled(Typography)(({ theme }) => ({
  fontSize: '0.875rem',
  fontWeight: 500,
  color: theme.palette.text.secondary,
  marginBottom: theme.spacing(1),
}));

const ChartPaper = styled(Box)(({ theme }) => ({
  padding: theme.spacing(3),
  paddingLeft: theme.spacing(6),
  paddingBottom: theme.spacing(5),
  background: 'transparent',
  backdropFilter: 'blur(10px)',
  border: 'none',
  boxShadow: 'none',
  borderRadius: theme.shape.borderRadius * 3,
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-2px)',
  },
  '& .recharts-bar-rectangle:hover': {
    fill: 'inherit !important',
  },
  '& .recharts-active-bar': {
    fill: 'inherit !important',
  }
}));

// Modern vibrant color palette
const CHART_COLORS = {
  primary: '#3b82f6', // Bright Blue
  secondary: '#10b981', // Emerald
  tertiary: '#f59e0b', // Amber
  quaternary: '#8b5cf6', // Violet
  quinary: '#ef4444', // Red
  senary: '#06b6d4', // Cyan
  pink: '#ec4899', // Pink
  indigo: '#6366f1', // Indigo
  teal: '#14b8a6', // Teal
  orange: '#f97316', // Orange
};

const PIE_COLORS = [
  CHART_COLORS.primary,
  CHART_COLORS.secondary,
  CHART_COLORS.tertiary,
  CHART_COLORS.quaternary,
  CHART_COLORS.pink,
  CHART_COLORS.indigo,
];

// Custom hook for animating numbers
const useAnimatedNumber = (value, duration = 1500) => {
  const [displayValue, setDisplayValue] = useState(0);
  
  useEffect(() => {
    const startTime = Date.now();
    const startValue = 0;
    const endValue = value;
    
    const updateNumber = () => {
      const now = Date.now();
      const progress = Math.min((now - startTime) / duration, 1);
      
      // Easing function
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      const currentValue = Math.floor(startValue + (endValue - startValue) * easeOutQuart);
      
      setDisplayValue(currentValue);
      
      if (progress < 1) {
        requestAnimationFrame(updateNumber);
      }
    };
    
    requestAnimationFrame(updateNumber);
  }, [value, duration]);
  
  return displayValue;
};

// Custom reference line for charts
const CustomReferenceLine = ({ x, y, stroke }) => {
  if (!x || !y) return null;
  
  return (
    <line
      x1={x}
      y1={y.min}
      x2={x}
      y2={y.max}
      stroke={stroke}
      strokeWidth={2}
      strokeOpacity={0.3}
      strokeDasharray="5 5"
    />
  );
};

// Custom active shape for pie chart
const renderActiveShape = (props) => {
  const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill, payload, value, percent } = props;

  return (
    <g>
      <text x={cx} y={cy - 10} dy={8} textAnchor="middle" fill={fill} fontSize="14" fontWeight="600">
        {payload.class_name}
      </text>
      <text x={cx} y={cy + 10} dy={8} textAnchor="middle" fill="#666" fontSize="12">
        {`${value} (${(percent * 100).toFixed(1)}%)`}
      </text>
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius + 10}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
      />
    </g>
  );
};

const Dashboard = () => {
  const theme = useTheme();
  const [selectedModel, setSelectedModel] = useState('all');
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [activeIndex, setActiveIndex] = useState(0);
  const [animationKey, setAnimationKey] = useState(0);
  
  // State for different data sections
  const [dailyData, setDailyData] = useState([]);
  const [weeklyData, setWeeklyData] = useState([]);
  const [stats, setStats] = useState({
    totalDetections: 0,
    objectDetections: 0,
    segmentations: 0,
    poseEstimations: 0
  });
  const [realTimeStats, setRealTimeStats] = useState({
    detectionRate: [],
    activeCameras: 0,
    latestDetections: []
  });
  const [topClasses, setTopClasses] = useState([]);
  const [cameraPerformance, setCameraPerformance] = useState([]);
  const [hourlyPattern, setHourlyPattern] = useState([]);
  const [selectedClass, setSelectedClass] = useState('all');
  const [classOptions, setClassOptions] = useState([]);
  
  // Animated values
  const animatedTotal = useAnimatedNumber(stats.totalDetections);
  const animatedObjects = useAnimatedNumber(stats.objectDetections);
  const animatedRate = useAnimatedNumber(realTimeStats.detectionRate.length > 0 ? realTimeStats.detectionRate[0].count : 0);

  const handleModelChange = (event, newModel) => {
    if (newModel !== null) {
      setSelectedModel(newModel);
    }
  };

  // Fetch all data
  const fetchAllData = async () => {
    try {
      setLoading(true);
      const [
        dailyRes,
        weeklyRes,
        statsRes,
        realTimeRes,
        topClassesRes,
        cameraRes,
        hourlyRes,
        classesRes
      ] = await Promise.allSettled([
        axios.get(`http://localhost:8000/api/detection-stats/daily?model=${selectedModel}`),
        axios.get(`http://localhost:8000/api/detection-stats/weekly?model=${selectedModel}`),
        axios.get('http://localhost:8000/api/detection-stats/summary'),
        axios.get('http://localhost:8000/api/detection-stats/real-time'),
        axios.get('http://localhost:8000/api/detection-stats/top-classes'),
        axios.get('http://localhost:8000/api/detection-stats/camera-performance'),
        axios.get('http://localhost:8000/api/detection-stats/hourly-pattern'),
        axios.get(`http://localhost:8000/api/detection-stats/classes?model=${selectedModel}`)
      ]);

      // Process daily data
      if (dailyRes.status === 'fulfilled') {
        setDailyData(dailyRes.value.data.map(item => ({
          ...item,
          hour: format(parseISO(item.timestamp), 'HH:mm'),
          count: parseInt(item.count)
        })));
      } else {
        console.error('Error fetching daily data:', dailyRes.reason);
        setDailyData([]);
      }

      // Process weekly data
      if (weeklyRes.status === 'fulfilled') {
        setWeeklyData(weeklyRes.value.data.map(item => ({
          ...item,
          date: format(parseISO(item.date), 'MMM dd'),
          count: parseInt(item.count)
        })));
      } else {
        console.error('Error fetching weekly data:', weeklyRes.reason);
        setWeeklyData([]);
      }

      if (statsRes.status === 'fulfilled') {
        setStats(statsRes.value.data);
      } else {
        console.error('Error fetching stats:', statsRes.reason);
        setStats({ totalDetections: 0, objectDetections: 0, segmentations: 0, poseEstimations: 0 });
      }

      if (realTimeRes.status === 'fulfilled') {
        setRealTimeStats(realTimeRes.value.data);
      } else {
        console.error('Error fetching real-time stats:', realTimeRes.reason);
        setRealTimeStats({ detectionRate: [], activeCameras: 0, latestDetections: [] });
      }
      
      // Calculate percentages for pie chart
      if (topClassesRes.status === 'fulfilled') {
        const total = topClassesRes.value.data.reduce((sum, item) => sum + item.count, 0);
        setTopClasses(topClassesRes.value.data.map(item => ({
          ...item,
          percentage: total > 0 ? ((item.count / total) * 100).toFixed(1) : '0'
        })));
      } else {
        console.error('Error fetching top classes:', topClassesRes.reason);
        setTopClasses([]);
      }

      if (cameraRes.status === 'fulfilled') {
        setCameraPerformance(cameraRes.value.data);
      } else {
        console.error('Error fetching camera performance:', cameraRes.reason);
        setCameraPerformance([]);
      }

      if (hourlyRes.status === 'fulfilled') {
        setHourlyPattern(hourlyRes.value.data);
      } else {
        console.error('Error fetching hourly pattern:', hourlyRes.reason);
        setHourlyPattern([]);
      }

      if (classesRes.status === 'fulfilled') {
        setClassOptions(classesRes.value.data);
      } else {
        console.error('Error fetching classes:', classesRes.reason);
        setClassOptions([]);
      }
      
      // Trigger animation on data load
      setAnimationKey(prev => prev + 1);
    } catch (error) {
      console.error('Unexpected error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllData();
  }, [selectedModel]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchAllData, 30000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, selectedModel]);

  // Calculate trend
  const calculateTrend = (data) => {
    if (data.length < 2) return { value: 0, isUp: true };
    const recent = data.slice(-2);
    const diff = recent[1].count - recent[0].count;
    const percentage = recent[0].count ? ((diff / recent[0].count) * 100).toFixed(1) : 0;
    return { value: Math.abs(percentage), isUp: diff >= 0 };
  };

  const trend = calculateTrend(dailyData);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  const onPieEnter = (_, index) => {
    setActiveIndex(index);
  };

  return (
    <Box sx={{ 
      p: 0,
      minHeight: '100vh',
      background: theme => theme.palette.mode === 'dark' 
        ? 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%)'
        : 'linear-gradient(135deg, #fafbfc 0%, #f6f8fa 50%, #f0f2f5 100%)',
      position: 'relative',
      '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: theme => theme.palette.mode === 'dark'
          ? 'radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%), radial-gradient(circle at 20% 80%, rgba(6, 182, 212, 0.1) 0%, transparent 50%)'
          : 'radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.08) 0%, transparent 50%), radial-gradient(circle at 20% 80%, rgba(6, 182, 212, 0.08) 0%, transparent 50%)',
        pointerEvents: 'none',
      },
      '& .recharts-wrapper': {
        '& .recharts-surface': {
          background: 'transparent !important',
        }
      },
      '& .recharts-tooltip-wrapper': {
        zIndex: 1000,
      },
      '& .recharts-bar': {
        cursor: 'pointer',
      },
      '& .recharts-bar-background-rectangle': {
        fill: 'transparent !important',
        display: 'none !important',
      },
      '& .recharts-bar-rectangle': {
        '&:hover': {
          fill: 'inherit !important',
          filter: 'inherit !important',
          opacity: '1 !important',
        }
      },
      '& .recharts-active-bar': {
        fill: 'inherit !important',
        opacity: '1 !important',
      },
      '& .recharts-bar-rectangles': {
        '& .recharts-bar-rectangle:not(:first-child)': {
          display: 'none !important',
        }
      }
    }}>
      <Box sx={{ p: 3, position: 'relative', zIndex: 1 }}>
        {/* Header */}
        <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        mb: 4 
      }}>
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Typography
            variant="h4"
            sx={{
              fontWeight: 600,
              color: theme.palette.text.primary,
              letterSpacing: '-0.02em',
            }}
          >
            Detection Analytics
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            Real-time monitoring dashboard
          </Typography>
        </motion.div>

        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Tooltip title={autoRefresh ? "Auto-refresh enabled" : "Auto-refresh disabled"}>
            <IconButton 
              onClick={() => setAutoRefresh(!autoRefresh)}
              sx={{ 
                bgcolor: autoRefresh ? alpha(CHART_COLORS.primary, 0.1) : 'transparent',
                color: autoRefresh ? CHART_COLORS.primary : 'text.secondary',
                '&:hover': {
                  bgcolor: autoRefresh ? alpha(CHART_COLORS.primary, 0.2) : alpha(theme.palette.action.hover, 0.1),
                }
              }}
            >
              <ArrowClockwise size={20} weight={autoRefresh ? "fill" : "regular"} />
            </IconButton>
          </Tooltip>
          
          <StyledToggleButtonGroup
            value={selectedModel}
            exclusive
            onChange={handleModelChange}
            aria-label="model selection"
          >
            <ToggleButton value="all">All Models</ToggleButton>
            <ToggleButton value="objectDetection">Objects</ToggleButton>
            <ToggleButton value="segmentation">Segmentation</ToggleButton>
            <ToggleButton value="pose">Pose</ToggleButton>
          </StyledToggleButtonGroup>
        </Box>
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <StatCard>
              <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Box>
                      <MetricLabel>Total Detections</MetricLabel>
                      <MetricValue>{animatedTotal.toLocaleString()}</MetricValue>
                    </Box>
                    <Box sx={{ 
                      width: 40, 
                      height: 40, 
                      borderRadius: 2,
                      bgcolor: alpha(CHART_COLORS.primary, 0.1),
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      <Pulse size={20} color={CHART_COLORS.primary} weight="bold" />
                    </Box>
                  </Box>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ 
                    display: 'flex', 
                    alignItems: 'center',
                    color: trend.isUp ? CHART_COLORS.secondary : CHART_COLORS.quinary,
                  }}>
                    {trend.isUp ? (
                      <CaretUp size={16} weight="fill" />
                    ) : (
                      <CaretDown size={16} weight="fill" />
                    )}
                    <Typography variant="body2" fontWeight={600}>
                      {trend.value}%
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    vs last hour
                  </Typography>
                </Box>
              </CardContent>
            </StatCard>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <StatCard>
              <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Box>
                      <MetricLabel>Active Cameras</MetricLabel>
                      <MetricValue>{realTimeStats.activeCameras}/{cameraPerformance.length}</MetricValue>
                    </Box>
                    <Box sx={{ 
                      width: 40, 
                      height: 40, 
                      borderRadius: 2,
                      bgcolor: alpha(CHART_COLORS.secondary, 0.1),
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      <CameraIcon size={20} color={CHART_COLORS.secondary} weight="bold" />
                    </Box>
                  </Box>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={(realTimeStats.activeCameras / cameraPerformance.length) * 100 || 0}
                  sx={{ 
                    height: 6,
                    borderRadius: 3,
                    bgcolor: alpha(CHART_COLORS.secondary, 0.1),
                    '& .MuiLinearProgress-bar': { 
                      bgcolor: CHART_COLORS.secondary,
                      borderRadius: 3,
                    }
                  }}
                />
              </CardContent>
            </StatCard>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <StatCard>
              <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Box>
                      <MetricLabel>Detection Rate</MetricLabel>
                      <MetricValue>
                        {animatedRate}
                        <Typography component="span" variant="body1" color="text.secondary" sx={{ ml: 0.5 }}>
                          /min
                        </Typography>
                      </MetricValue>
                    </Box>
                    <Box sx={{ 
                      width: 40, 
                      height: 40, 
                      borderRadius: 2,
                      bgcolor: alpha(CHART_COLORS.tertiary, 0.1),
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      <Lightning size={20} color={CHART_COLORS.tertiary} weight="bold" />
                    </Box>
                  </Box>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  Current detection frequency
                </Typography>
              </CardContent>
            </StatCard>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <StatCard>
              <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Box>
                      <MetricLabel>Object Detections</MetricLabel>
                      <MetricValue>{animatedObjects.toLocaleString()}</MetricValue>
                    </Box>
                    <Box sx={{ 
                      width: 40, 
                      height: 40, 
                      borderRadius: 2,
                      bgcolor: alpha(CHART_COLORS.quaternary, 0.1),
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      <Eye size={20} color={CHART_COLORS.quaternary} weight="bold" />
                    </Box>
                  </Box>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  {((stats.objectDetections / stats.totalDetections) * 100).toFixed(1)}% of total
                </Typography>
              </CardContent>
            </StatCard>
          </motion.div>
        </Grid>
      </Grid>

      {/* Charts Row 1 */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {/* Real-time Activity */}
        <Grid item xs={12} lg={8}>
          <ChartPaper>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Box>
                <Typography variant="h6" fontWeight={600}>
                  Today's Detections (Last 24 Hours)
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                  Number of detections per hour
                </Typography>
              </Box>
              {selectedModel !== 'all' && classOptions.length > 0 && (
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  <Chip
                    size="small"
                    label="All"
                    onClick={() => setSelectedClass('all')}
                    color={selectedClass === 'all' ? 'primary' : 'default'}
                    variant={selectedClass === 'all' ? 'filled' : 'outlined'}
                    sx={{ borderRadius: 2 }}
                  />
                  {classOptions.slice(0, 4).map((className) => (
                    <Chip
                      key={className}
                      size="small"
                      label={className}
                      onClick={() => setSelectedClass(className)}
                      color={selectedClass === className ? 'primary' : 'default'}
                      variant={selectedClass === className ? 'filled' : 'outlined'}
                      sx={{ borderRadius: 2 }}
                    />
                  ))}
                </Stack>
              )}
            </Box>
            <Box sx={{ height: 300 }}>
              {dailyData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={dailyData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <defs>
                    <linearGradient id="lineGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CHART_COLORS.teal} stopOpacity={0.8}/>
                      <stop offset="100%" stopColor={CHART_COLORS.teal} stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="0" stroke="transparent" />
                  <XAxis 
                    dataKey="hour"
                    tick={{ fill: theme.palette.text.secondary, fontSize: 12 }}
                    axisLine={{ stroke: alpha(theme.palette.divider, 0.2) }}
                    label={{ value: 'Hour of Day', position: 'insideBottom', offset: -5, style: { fill: theme.palette.text.secondary, fontSize: 12 } }}
                  />
                  <YAxis 
                    tick={{ fill: theme.palette.text.secondary, fontSize: 12 }}
                    axisLine={{ stroke: alpha(theme.palette.divider, 0.2) }}
                    allowDecimals={false}
                    domain={[0, 'auto']}
                    label={{ 
                      value: 'Detection Count', 
                      angle: -90, 
                      position: 'insideLeft',
                      style: { 
                        textAnchor: 'middle',
                        fill: theme.palette.text.secondary,
                        fontSize: 12
                      }
                    }}
                  />
                  <ChartTooltip 
                    wrapperStyle={{ 
                      outline: 'none'
                    }}
                    labelFormatter={(hour) => `Time: ${hour}:00 - ${hour}:59`}
                    formatter={(value) => {
                      if (value === 0) return ['No detections'];
                      return [
                        `Total: ${value} detections`,
                        `Most common: ${topClasses.length > 0 ? topClasses[0].class_name : 'N/A'}`,
                        `Active cameras: ${realTimeStats.activeCameras}`
                      ];
                    }}
                    cursor={false}
                    contentStyle={{ 
                      backgroundColor: theme.palette.mode === 'dark' 
                        ? 'rgba(20, 20, 20, 0.95)' 
                        : 'rgba(248, 249, 250, 0.98)',
                      border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                      borderRadius: 8,
                      boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
                      backdropFilter: 'blur(10px)',
                      padding: '16px',
                      minWidth: '200px'
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="count"
                    stroke={CHART_COLORS.teal}
                    strokeWidth={4}
                    dot={{ fill: CHART_COLORS.teal, r: 6, strokeWidth: 2, stroke: '#fff' }}
                    activeDot={{ r: 8, strokeWidth: 2, stroke: '#fff' }}
                    isAnimationActive={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="count"
                    stroke="none"
                    fill="url(#lineGradient)"
                    fillOpacity={0.3}
                    isAnimationActive={false}
                  />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <Typography color="text.secondary">No data available</Typography>
                </Box>
              )}
            </Box>
          </ChartPaper>
        </Grid>

        {/* Top Classes */}
        <Grid item xs={12} lg={4}>
          <ChartPaper>
            <Typography variant="h6" fontWeight={600} sx={{ mb: 3 }}>
              Top Detected Classes
            </Typography>
            <Box sx={{ height: 300 }}>
              {topClasses.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      activeIndex={activeIndex}
                      activeShape={renderActiveShape}
                      data={topClasses.slice(0, 6)}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="count"
                      onMouseEnter={onPieEnter}
                      animationBegin={0}
                      animationDuration={1500}
                      animationEasing="ease-out"
                    >
                      {topClasses.slice(0, 6).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <Typography color="text.secondary">No data available</Typography>
                </Box>
              )}
            </Box>
          </ChartPaper>
        </Grid>
      </Grid>

      {/* Charts Row 2 */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {/* Weekly Trend */}
        <Grid item xs={12} lg={6}>
          <ChartPaper>
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" fontWeight={600}>
                This Week's Detections
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                Daily totals for the past 7 days
              </Typography>
            </Box>
            <Box sx={{ height: 300 }}>
              {weeklyData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={weeklyData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <defs>
                    <linearGradient id="weeklyLineGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CHART_COLORS.pink} stopOpacity={0.8}/>
                      <stop offset="100%" stopColor={CHART_COLORS.pink} stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="0" stroke="transparent" />
                  <XAxis 
                    dataKey="date"
                    tick={{ fill: theme.palette.text.secondary, fontSize: 12 }}
                    axisLine={{ stroke: alpha(theme.palette.divider, 0.2) }}
                    label={{ value: 'Day', position: 'insideBottom', offset: -5, style: { fill: theme.palette.text.secondary, fontSize: 12 } }}
                  />
                  <YAxis 
                    tick={{ fill: theme.palette.text.secondary, fontSize: 12 }}
                    axisLine={{ stroke: alpha(theme.palette.divider, 0.2) }}
                    allowDecimals={false}
                    domain={[0, 'auto']}
                    label={{ 
                      value: 'Total Detections', 
                      angle: -90, 
                      position: 'insideLeft',
                      style: { 
                        textAnchor: 'middle',
                        fill: theme.palette.text.secondary,
                        fontSize: 12
                      }
                    }}
                  />
                  <ChartTooltip 
                    wrapperStyle={{ 
                      outline: 'none'
                    }}
                    formatter={(value, name, props) => {
                      const dayData = props.payload;
                      if (value === 0) return [`No detections on ${dayData.date}`];
                      const avgPerHour = Math.round(value / 24);
                      return [
                        `Total: ${value} detections`,
                        `Date: ${dayData.date}`,
                        `Average per hour: ${avgPerHour}`,
                        `Peak detection types: ${topClasses.slice(0, 3).map(c => c.class_name).join(', ') || 'N/A'}`
                      ];
                    }}
                    cursor={false}
                    contentStyle={{ 
                      backgroundColor: theme.palette.mode === 'dark' 
                        ? 'rgba(20, 20, 20, 0.95)' 
                        : 'rgba(248, 249, 250, 0.98)',
                      border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                      borderRadius: 8,
                      boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
                      backdropFilter: 'blur(10px)',
                      padding: '16px',
                      minWidth: '200px'
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="count"
                    stroke={CHART_COLORS.pink}
                    strokeWidth={4}
                    dot={{ fill: CHART_COLORS.pink, r: 6, strokeWidth: 2, stroke: '#fff' }}
                    activeDot={{ r: 8, strokeWidth: 2, stroke: '#fff' }}
                    isAnimationActive={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="count"
                    stroke="none"
                    fill="url(#weeklyLineGradient)"
                    fillOpacity={0.3}
                    isAnimationActive={false}
                  />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <Typography color="text.secondary">No data available</Typography>
                </Box>
              )}
            </Box>
          </ChartPaper>
        </Grid>

        {/* Hourly Pattern */}
        <Grid item xs={12} lg={6}>
          <ChartPaper>
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" fontWeight={600}>
                Typical Day Pattern
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                Average detections throughout a typical day (0-23 hours)
              </Typography>
            </Box>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={hourlyPattern}>
                  <CartesianGrid strokeDasharray="0" stroke="transparent" />
                  <XAxis 
                    dataKey="hour"
                    tickFormatter={(hour) => `${hour}:00`}
                    tick={{ fill: theme.palette.text.secondary, fontSize: 12 }}
                    axisLine={{ stroke: alpha(theme.palette.divider, 0.2) }}
                    label={{ value: 'Hour of Day', position: 'insideBottom', offset: -5, style: { fill: theme.palette.text.secondary, fontSize: 12 } }}
                  />
                  <YAxis 
                    tick={{ fill: theme.palette.text.secondary, fontSize: 12 }}
                    axisLine={{ stroke: alpha(theme.palette.divider, 0.2) }}
                    allowDecimals={false}
                    domain={[0, 'auto']}
                    tickFormatter={(value) => Math.round(value)}
                    label={{ 
                      value: 'Average Count', 
                      angle: -90, 
                      position: 'insideLeft',
                      style: { 
                        textAnchor: 'middle',
                        fill: theme.palette.text.secondary,
                        fontSize: 12
                      }
                    }}
                  />
                  <ChartTooltip 
                    wrapperStyle={{ 
                      outline: 'none'
                    }}
                    labelFormatter={(hour) => `${hour}:00 - ${hour}:59`}
                    formatter={(value) => {
                      const rounded = Math.round(value);
                      if (rounded === 0) return ['No typical detections at this hour'];
                      const activityLevel = rounded > 50 ? 'High activity' : rounded > 20 ? 'Moderate activity' : 'Low activity';
                      return [
                        `Average: ${rounded} detections per hour`,
                        `Activity level: ${activityLevel}`,
                        `Common objects: ${topClasses.slice(0, 2).map(c => c.class_name).join(', ') || 'Various'}`,
                        `Cameras typically active: ${Math.min(cameraPerformance.length, Math.ceil(rounded / 10))}`
                      ];
                    }}
                    cursor={false}
                    contentStyle={{ 
                      backgroundColor: theme.palette.mode === 'dark' 
                        ? 'rgba(20, 20, 20, 0.95)' 
                        : 'rgba(248, 249, 250, 0.98)',
                      border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                      borderRadius: 8,
                      boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
                      backdropFilter: 'blur(10px)',
                      padding: '16px',
                      minWidth: '200px'
                    }}
                  />
                  <defs>
                    <linearGradient id="lineGradient" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor={CHART_COLORS.teal} />
                      <stop offset="50%" stopColor={CHART_COLORS.secondary} />
                      <stop offset="100%" stopColor={CHART_COLORS.primary} />
                    </linearGradient>
                  </defs>
                  <defs>
                    <filter id="glowLine">
                      <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                      <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                      </feMerge>
                    </filter>
                  </defs>
                  <Line 
                    type="monotone" 
                    dataKey="avgCount" 
                    stroke="url(#lineGradient)"
                    strokeWidth={4}
                    dot={false}
                    activeDot={{ r: 4, fill: CHART_COLORS.secondary, strokeWidth: 0 }}
                    isAnimationActive={true}
                    animationBegin={0}
                    animationDuration={3000}
                    animationEasing="ease-out"
                  />
                  </LineChart>
                </ResponsiveContainer>
            </Box>
          </ChartPaper>
        </Grid>
      </Grid>

      {/* Camera Performance Table */}
      <ChartPaper>
        <Typography variant="h6" fontWeight={600} sx={{ mb: 3 }}>
          Camera Performance
        </Typography>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Camera</TableCell>
                <TableCell>Location</TableCell>
                <TableCell align="center">Status</TableCell>
                <TableCell align="right">Detections</TableCell>
                <TableCell align="right">Last Active</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {cameraPerformance.map((camera) => (
                <TableRow 
                  key={camera.id}
                  sx={{ '&:hover': { bgcolor: alpha(theme.palette.action.hover, 0.04) } }}
                >
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                      <Avatar 
                        sx={{ 
                          width: 36, 
                          height: 36, 
                          background: `linear-gradient(135deg, ${CHART_COLORS.primary}, ${CHART_COLORS.indigo})`,
                          color: 'white',
                          fontSize: '0.875rem',
                          fontWeight: 600,
                          boxShadow: `0 4px 12px ${alpha(CHART_COLORS.primary, 0.3)}`
                        }}
                      >
                        {camera.name.charAt(0).toUpperCase()}
                      </Avatar>
                      <Typography variant="body2" fontWeight={500}>
                        {camera.name}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" color="text.secondary">
                      {camera.location || 'Not specified'}
                    </Typography>
                  </TableCell>
                  <TableCell align="center">
                    <Chip
                      size="small"
                      label={camera.isActive ? 'Active' : 'Inactive'}
                      sx={{
                        background: camera.isActive 
                          ? `linear-gradient(135deg, ${alpha(CHART_COLORS.secondary, 0.2)}, ${alpha(CHART_COLORS.teal, 0.2)})` 
                          : alpha(theme.palette.text.secondary, 0.1),
                        color: camera.isActive 
                          ? CHART_COLORS.secondary 
                          : theme.palette.text.secondary,
                        border: 'none',
                        fontWeight: 600,
                        backdropFilter: 'blur(10px)',
                      }}
                    />
                  </TableCell>
                  <TableCell align="right">
                    <Typography variant="body2" fontWeight={500}>
                      {camera.detectionCount.toLocaleString()}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Typography variant="body2" color="text.secondary">
                      {camera.lastDetection 
                        ? formatDistanceToNow(parseISO(camera.lastDetection), { addSuffix: true })
                        : 'Never'}
                    </Typography>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </ChartPaper>
      </Box>
    </Box>
  );
};

export default Dashboard;