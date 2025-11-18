import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import {
    Box,
    Container,
    Typography,
    Radio,
    RadioGroup,
    FormControlLabel,
    Button,
    Paper,
    Card,
    CardContent,
    Grid,
    Fade,
    CircularProgress,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    IconButton,
    Chip,
    ToggleButton,
    ToggleButtonGroup,
    Divider,
    Switch,
    FormGroup,
    List,
    ListItem,
    ListItemText,
    Avatar,
    TextField,
    Stack,
    Tooltip,
} from '@mui/material';
import { styled, alpha, useTheme } from '@mui/material/styles';
import { 
    PlayArrow, 
    Stop, 
    Visibility, 
    PersonPin, 
    CropFree,
    CompareArrows,
    Speed,
    CheckCircle
} from '@mui/icons-material';
import { 
    Camera, 
    Play, 
    Pause, 
    Eye, 
    User, 
    Scan,
    Trash,
    ArrowCounterClockwise,
    ArrowClockwise,
    ChartLine,
    Timer as TimerIcon,
    UsersThree
} from '@phosphor-icons/react';
import { motion, AnimatePresence } from 'framer-motion';

// Styled components
const StyledContainer = styled(Container)(({ theme }) => ({
    paddingTop: theme.spacing(4),
    paddingBottom: theme.spacing(4),
    position: 'relative',
    zIndex: 1,
}));

const VideoContainer = styled(Paper)(({ theme }) => ({
    aspectRatio: '16/9',
    width: '100%',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    borderRadius: theme.spacing(3),
    background: theme.palette.mode === 'dark'
        ? alpha(theme.palette.background.paper, 0.03)
        : 'rgba(255, 255, 255, 0.7)',
    backdropFilter: 'blur(20px)',
    border: `1px solid ${alpha(theme.palette.divider, theme.palette.mode === 'dark' ? 0.08 : 0.12)}`,
    boxShadow: 'none',
    padding: 0,
    position: 'relative',
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    '&:hover': {
        borderColor: alpha('#06b6d4', 0.3),
        boxShadow: `0 8px 32px ${alpha('#06b6d4', 0.1)}`,
    },
}));

const ControlButton = styled(Button)(({ theme }) => ({
    margin: theme.spacing(1),
    padding: theme.spacing(1, 3),
    borderRadius: theme.spacing(2),
}));


// Modern button styled component
const ModernButton = styled(Button)(({ theme, variant }) => ({
    borderRadius: '8px',
    padding: '10px 28px',
    textTransform: 'none',
    fontWeight: 600,
    fontSize: '0.95rem',
    background: variant === 'stop' 
        ? theme.palette.mode === 'dark'
            ? 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)'
            : 'linear-gradient(135deg, #f87171 0%, #ef4444 100%)'
        : theme.palette.mode === 'dark'
            ? 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)'
            : 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
    color: 'white',
    boxShadow: variant === 'stop'
        ? '0 4px 20px rgba(239, 68, 68, 0.3)'
        : '0 4px 20px rgba(6, 182, 212, 0.3)',
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    '&:hover': {
        background: variant === 'stop'
            ? theme.palette.mode === 'dark'
                ? 'linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)'
                : 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)'
            : theme.palette.mode === 'dark'
                ? 'linear-gradient(135deg, #0891b2 0%, #0e7490 100%)'
                : 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
        boxShadow: variant === 'stop'
            ? '0 6px 24px rgba(239, 68, 68, 0.4)'
            : '0 6px 24px rgba(6, 182, 212, 0.4)',
        transform: 'translateY(-2px)'
    },
    '&:disabled': {
        background: alpha(theme.palette.action.disabled, 0.12),
        color: theme.palette.action.disabled,
        boxShadow: 'none'
    }
}));

// Model selection button component
const ModelButton = styled(Button)(({ theme, selected, modelColor }) => ({
    cursor: 'pointer',
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    border: selected ? 'none' : `2px solid ${alpha(theme.palette.divider, 0.2)}`,
    borderRadius: '12px',
    padding: '14px 28px',
    minWidth: '140px',
    background: selected 
        ? `linear-gradient(135deg, ${modelColor} 0%, ${alpha(modelColor, 0.8)} 100%)`
        : theme.palette.mode === 'dark' 
            ? alpha(theme.palette.background.paper, 0.05)
            : 'rgba(255, 255, 255, 0.8)',
    color: selected 
        ? 'white' 
        : theme.palette.text.primary,
    boxShadow: selected 
        ? `0 8px 32px ${alpha(modelColor, 0.4)}`
        : 'none',
    '&:hover': {
        transform: 'translateY(-2px)',
        background: selected 
            ? `linear-gradient(135deg, ${alpha(modelColor, 0.9)} 0%, ${alpha(modelColor, 0.7)} 100%)`
            : theme.palette.mode === 'dark' 
                ? alpha(modelColor, 0.1)
                : alpha(modelColor, 0.08),
        borderColor: selected ? 'transparent' : alpha(modelColor, 0.3),
        boxShadow: selected 
            ? `0 12px 40px ${alpha(modelColor, 0.5)}`
            : `0 4px 20px ${alpha(modelColor, 0.15)}`,
    },
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '8px',
    textTransform: 'none',
    fontWeight: 600,
    position: 'relative',
    overflow: 'hidden',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: selected
            ? 'linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%)'
            : 'none',
        transform: 'translateX(-100%)',
        transition: 'transform 0.6s',
    },
    '&:hover::before': {
        transform: 'translateX(100%)',
    },
}));

// Model info data
const modelInfo = {
    objectDetection: {
        title: "Detection",
        icon: <Scan size={24} weight="duotone" />,
        color: '#06b6d4', // Cyan theme
    },
    segmentation: {
        title: "Segmentation",
        icon: <Eye size={24} weight="duotone" />,
        color: '#10b981', // Emerald
    },
    pose: {
        title: "Pose",
        icon: <User size={24} weight="duotone" />,
        color: '#f59e0b', // Amber
    }
};

// Split view component for multiple models
const SplitViewContainer = styled(Box)(({ theme }) => ({
    display: 'grid',
    gap: theme.spacing(2),
    height: '100%',
    gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
}));

// Main component
const LiveAnalysis = () => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
    const [cameras, setCameras] = useState([]);
    const [selectedModel, setSelectedModel] = useState('objectDetection');
    
    // Add this state to track if we're actively streaming
    const [isStreaming, setIsStreaming] = useState(false);
    const [selectedCameraId, setSelectedCameraId] = useState(null);
    const [zones, setZones] = useState([]);
    const [liveZoneStats, setLiveZoneStats] = useState({});
    const [drawingMode, setDrawingMode] = useState(false);
    const [draftPolygon, setDraftPolygon] = useState([]);
    const [newZoneName, setNewZoneName] = useState('');
    const [activeZoneId, setActiveZoneId] = useState(null);
    const [activeZoneStats, setActiveZoneStats] = useState(null);
    const [frameMeta, setFrameMeta] = useState({});
    const videoRefs = useRef({});

    const API_BASE = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

    const fetchZones = useCallback(async () => {
        if (!selectedCameraId) {
            setZones([]);
            return;
        }
        try {
            const { data } = await axios.get(`${API_BASE}/zones`, {
                params: { camera_id: selectedCameraId }
            });
            setZones(data);
            if (activeZoneId && !data.find((zone) => zone.zone_id === activeZoneId)) {
                setActiveZoneId(null);
            }
        } catch (error) {
            console.error('Error fetching zones', error);
        }
    }, [API_BASE, selectedCameraId, activeZoneId]);

    const fetchLiveZones = useCallback(async () => {
        try {
            const { data } = await axios.get(`${API_BASE}/zones/live`);
            const liveMap = {};
            data.forEach((item) => {
                liveMap[item.zone.zone_id] = item.live;
            });
            setLiveZoneStats(liveMap);
        } catch (error) {
            console.error('Error fetching live zone stats', error);
        }
    }, [API_BASE]);

    // Modify the useEffect to check existing streams when mounting
    useEffect(() => {
                const checkExistingStreams = async () => {
            try {
                // Fetch cameras first
                const response = await axios.get(`${API_BASE}/api/cameras`);
                // Include both local 'live' cameras and RTSP cameras
                const liveCameras = response.data.filter(camera => {
                    const st = (camera.stream_type || '').toLowerCase();
                    return st === 'live' || st === 'rtsp';
                });
                
                // Check if any streams are active by attempting to fetch frames
                const camerasWithStreamStatus = await Promise.all(liveCameras.map(async (camera) => {
                    try {
                        const frameResponse = await axios.get(
                            `${API_BASE}/process_frame/${camera.id}`,
                            { responseType: 'blob' }
                        );
                        // If we get a successful response, the stream is active
                        const isActive = frameResponse.status === 200;
                        return {
                            ...camera,
                            isStreaming: isActive,
                            videoSrc: isActive ? URL.createObjectURL(frameResponse.data) : '',
                        };
                    } catch (error) {
                        return {
                            ...camera,
                            isStreaming: false,
                            videoSrc: '',
                        };
                    }
                }));

                setCameras(camerasWithStreamStatus);
                setIsStreaming(camerasWithStreamStatus.some(cam => cam.isStreaming));
            } catch (error) {
                console.error("Error checking existing streams:", error);
            }
        };

        checkExistingStreams();
    }, []);

    useEffect(() => {
        fetchZones();
    }, [fetchZones]);

    useEffect(() => {
        fetchLiveZones();
        const interval = setInterval(fetchLiveZones, 3000);
        return () => clearInterval(interval);
    }, [fetchLiveZones]);

    useEffect(() => {
        setDraftPolygon([]);
        setDrawingMode(false);
    }, [selectedCameraId]);

    useEffect(() => {
        if (!selectedCameraId && cameras.length > 0) {
            setSelectedCameraId(cameras[0].id);
        }
    }, [cameras, selectedCameraId]);

    useEffect(() => {
        if (!activeZoneId) {
            setActiveZoneStats(null);
            return;
        }
        const loadStats = async () => {
            try {
                const { data } = await axios.get(`${API_BASE}/zones/${activeZoneId}/stats`);
                setActiveZoneStats(data);
            } catch (error) {
                console.error('Error fetching zone stats', error);
            }
        };
        loadStats();
    }, [API_BASE, activeZoneId]);

    // Modify the frame fetching useEffect
    useEffect(() => {
        const intervals = {};
        
        // Create separate intervals for each streaming camera
        cameras.forEach(camera => {
            if (camera.isStreaming) {
                intervals[camera.id] = setInterval(async () => {
                    try {
                        const response = await axios.get(
                            `${API_BASE}/process_frame/${camera.id}`,
                            { responseType: 'blob' }
                        );
                        
                        const url = URL.createObjectURL(response.data);
                        
                        setCameras(prevCameras => {
                            return prevCameras.map(prevCamera => {
                                if (prevCamera.id === camera.id) {
                                    // Cleanup old URL
                                    if (prevCamera.videoSrc) {
                                        URL.revokeObjectURL(prevCamera.videoSrc);
                                    }
                                    return { ...prevCamera, videoSrc: url };
                                }
                                return prevCamera;
                            });
                        });
                    } catch (error) {
                        console.error(`Error fetching frame for camera ${camera.id}:`, error);
                    }
                }, 100); // 100ms interval for each camera
            }
        });
        
        // Cleanup function
        return () => {
            // Clear all intervals
            Object.values(intervals).forEach(interval => clearInterval(interval));
            
            // Cleanup URLs if stopping all streams
            if (!isStreaming) {
                cameras.forEach(camera => {
                    if (camera.videoSrc) {
                        URL.revokeObjectURL(camera.videoSrc);
                    }
                });
            }
        };
    }, [cameras, isStreaming]);

    const startWebcamStream = async () => {
        try {
            // Start all cameras with selected model
            await Promise.all(cameras.map(async (camera) => {
                await axios.post(`${API_BASE}/start_camera_stream/${camera.id}`, {
                    model_type: selectedModel,
                });
            }));
            
            const updatedCameras = cameras.map(camera => ({
                ...camera,
                isStreaming: true,
            }));
            
            setCameras(updatedCameras);
            setIsStreaming(true);
        } catch (error) {
            console.error('Error starting camera streams:', error);
        }
    };

    const stopWebcamStream = async () => {
        try {
            // Stop all cameras
            await Promise.all(cameras.map(async (camera) => {
            await axios.post(`${API_BASE}/stop_camera_stream/${camera.id}`);
            }));
            
            const updatedCameras = cameras.map(camera => ({
                ...camera,
                isStreaming: false,
                videoSrc: '',
            }));
            
            setCameras(updatedCameras);
            setIsStreaming(false);
        } catch (error) {
            console.error('Error stopping camera streams:', error);
        }
    };

    const zoneColorPalette = ['#FF5C8D', '#06b6d4', '#34d399', '#f97316', '#a855f7', '#f97316'];
    const pickZoneColor = () => zoneColorPalette[Math.floor(Math.random() * zoneColorPalette.length)];

    const startSingleCamera = async (cameraId) => {
        try {
            await axios.post(`${API_BASE}/start_camera_stream/${cameraId}`, {
                model_type: selectedModel,
            });
            
            const updatedCameras = cameras.map(camera => ({
                ...camera,
                isStreaming: camera.id === cameraId ? true : camera.isStreaming,
            }));
            
            setCameras(updatedCameras);
            setIsStreaming(true);
        } catch (error) {
            console.error(`Error starting camera ${cameraId} stream:`, error);
        }
    };

    const handleModelSelection = (modelType) => {
        setSelectedModel(modelType);
    };

    const stopSingleCamera = async (cameraId) => {
        try {
            await axios.post(`${API_BASE}/stop_camera_stream/${cameraId}`);
            
            const updatedCameras = cameras.map(camera => ({
                ...camera,
                isStreaming: camera.id === cameraId ? false : camera.isStreaming,
                videoSrc: camera.id === cameraId ? '' : camera.videoSrc,
            }));
            
            setCameras(updatedCameras);
            setIsStreaming(updatedCameras.some(cam => cam.isStreaming));
        } catch (error) {
            console.error(`Error stopping camera ${cameraId} stream:`, error);
        }
    };

    const handleFrameLoad = (cameraId, event) => {
        const { naturalWidth, naturalHeight } = event.target;
        if (!naturalWidth || !naturalHeight) {
            return;
        }
        setFrameMeta(prev => ({
            ...prev,
            [cameraId]: {
                width: naturalWidth,
                height: naturalHeight
            }
        }));
    };

    const handleStartDrawing = () => {
        if (!selectedCameraId) return;
        setDrawingMode(true);
        setDraftPolygon([]);
    };

    const handleCancelDrawing = () => {
        setDrawingMode(false);
        setDraftPolygon([]);
        setNewZoneName('');
    };

    const handleOverlayClick = (event, cameraId) => {
        if (!drawingMode || cameraId !== selectedCameraId) {
            return;
        }
        const bounds = event.currentTarget.getBoundingClientRect();
        const frame = frameMeta[cameraId];
        if (!frame) {
            return;
        }
        const relX = event.clientX - bounds.left;
        const relY = event.clientY - bounds.top;
        const x = Number(((relX / bounds.width) * frame.width).toFixed(2));
        const y = Number(((relY / bounds.height) * frame.height).toFixed(2));
        setDraftPolygon(prev => [...prev, [x, y]]);
    };

    const handleFinishDrawing = async () => {
        if (!selectedCameraId || draftPolygon.length < 3) {
            return;
        }
        const payload = {
            camera_id: selectedCameraId,
            name: newZoneName || `Zone ${zones.length + 1}`,
            polygon: draftPolygon,
            color: pickZoneColor(),
            attribution_mode: 'multiple',
            properties: { created_from: 'ui' }
        };
        try {
            await axios.post(`${API_BASE}/zones`, payload);
            setDrawingMode(false);
            setDraftPolygon([]);
            setNewZoneName('');
            fetchZones();
        } catch (error) {
            console.error('Error saving zone', error);
        }
    };

    const handleZoneDelete = async (zoneId) => {
        try {
            await axios.delete(`${API_BASE}/zones/${zoneId}`);
            if (activeZoneId === zoneId) {
                setActiveZoneId(null);
            }
            fetchZones();
        } catch (error) {
            console.error('Failed to delete zone', error);
        }
    };

    const handleUndo = async () => {
        try {
            await axios.post(`${API_BASE}/zones/undo`);
            fetchZones();
        } catch (error) {
            console.error('Nothing to undo', error);
        }
    };

    const handleRedo = async () => {
        try {
            await axios.post(`${API_BASE}/zones/redo`);
            fetchZones();
        } catch (error) {
            console.error('Nothing to redo', error);
        }
    };

    const handleZoneSelect = (zoneId) => {
        setActiveZoneId(zoneId);
    };

    const handleCameraSelection = (event) => {
        const value = event.target.value;
        setSelectedCameraId(value === '' ? null : Number(value));
        setActiveZoneId(null);
    };

    const getDraftPolylinePoints = (cameraId) => {
        const node = videoRefs.current[cameraId];
        const frame = frameMeta[cameraId];
        if (!node || !frame || draftPolygon.length === 0) {
            return '';
        }
        const rect = node.getBoundingClientRect();
        return draftPolygon
            .map(([x, y]) => {
                const px = (x / frame.width) * rect.width;
                const py = (y / frame.height) * rect.height;
                return `${px},${py}`;
            })
            .join(' ');
    };

    const activeLive = activeZoneId ? (liveZoneStats[activeZoneId] || { count: 0, objects: [] }) : null;

    return (
        <StyledContainer maxWidth="xl">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <Box sx={{ mb: 6, textAlign: 'center' }}>
                    <Typography 
                        variant="h3" 
                        sx={{ 
                            mb: 2,
                            fontWeight: 800,
                            background: isDark
                                ? 'linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)'
                                : 'linear-gradient(135deg, #0891b2 0%, #2563eb 100%)',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                        }}
                    >
                        Live Analysis
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                        Real-time object detection and analysis from your cameras
                    </Typography>
                </Box>

                {/* Model Selection */}
                <Box sx={{ mb: 4, display: 'flex', gap: 2, flexWrap: 'wrap', justifyContent: 'center' }}>
                        {Object.entries(modelInfo).map(([key, info]) => (
                            <motion.div
                                key={key}
                                whileHover={{ scale: 1.03 }}
                                whileTap={{ scale: 0.97 }}
                            >
                                <ModelButton 
                                    selected={selectedModel === key}
                                    onClick={() => handleModelSelection(key)}
                                    modelColor={info.color}
                                >
                                    <Box sx={{ 
                                        color: selectedModel === key ? 'white' : info.color,
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        mb: 0.5,
                                    }}>
                                        {React.cloneElement(info.icon, { 
                                            size: 28,
                                            weight: selectedModel === key ? 'fill' : 'duotone'
                                        })}
                                    </Box>
                                    <Typography variant="body1" sx={{ 
                                        fontSize: '0.95rem',
                                        fontWeight: selectedModel === key ? 600 : 500,
                                    }}>
                                        {info.title}
                                    </Typography>
                                </ModelButton>
                            </motion.div>
                        ))}
                </Box>

                {/* Control Buttons */}
                <Box sx={{ 
                    display: 'flex',
                    justifyContent: 'center',
                    gap: 2,
                    mb: 4
                }}>
                    <ModernButton
                        disabled={cameras.some(camera => camera.isStreaming)}
                        onClick={startWebcamStream}
                        startIcon={<Play size={20} weight="bold" />}
                        size="large"
                    >
                        Start All Cameras
                    </ModernButton>
                    <ModernButton
                        variant="stop"
                        disabled={!cameras.some(camera => camera.isStreaming)}
                        onClick={stopWebcamStream}
                        startIcon={<Pause size={20} weight="bold" />}
                        size="large"
                    >
                        Stop All
                    </ModernButton>
                </Box>


                <Grid container spacing={3}>
                    <Grid item xs={12} md={4}>
                        {/* Zone control panel */}
                        <Paper sx={{
                            p: 3,
                            borderRadius: 3,
                            background: isDark ? alpha(theme.palette.background.paper, 0.08) : alpha('#ffffff', 0.95),
                            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`
                        }}>
                            <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>
                                Zone Controls
                            </Typography>
                            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                                <InputLabel id="zone-camera-select">Camera</InputLabel>
                                <Select
                                    labelId="zone-camera-select"
                                    value={selectedCameraId ?? ''}
                                    label="Camera"
                                    onChange={handleCameraSelection}
                                >
                                    <MenuItem value="" disabled>
                                        Select camera
                                    </MenuItem>
                                    {cameras.map((camera) => (
                                        <MenuItem key={camera.id} value={camera.id}>
                                            {camera.source_name || `Camera ${camera.id}`}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                            <TextField
                                label="Zone name"
                                size="small"
                                fullWidth
                                value={newZoneName}
                                onChange={(event) => setNewZoneName(event.target.value)}
                                sx={{ mb: 2 }}
                            />
                            <Stack spacing={1}>
                                <Stack direction="row" spacing={1}>
                                    <Button variant="outlined" onClick={handleStartDrawing} disabled={!selectedCameraId}>
                                        Draw
                                    </Button>
                                    <Button variant="contained" onClick={handleFinishDrawing} disabled={draftPolygon.length < 3}>
                                        Save
                                    </Button>
                                    <Button variant="text" onClick={handleCancelDrawing} disabled={!drawingMode && draftPolygon.length === 0}>
                                        Cancel
                                    </Button>
                                </Stack>
                                <Stack direction="row" spacing={1}>
                                    <Button startIcon={<ArrowCounterClockwise size={16} />} onClick={handleUndo}>
                                        Undo
                                    </Button>
                                    <Button startIcon={<ArrowClockwise size={16} />} onClick={handleRedo}>
                                        Redo
                                    </Button>
                                </Stack>
                            </Stack>
                            {drawingMode && (
                                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                                    Click on the live feed to place vertices ({draftPolygon.length})
                                </Typography>
                            )}
                            <Divider sx={{ my: 2 }} />
                            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                                Zones ({zones.length})
                            </Typography>
                            {zones.length === 0 ? (
                                <Typography variant="body2" color="text.secondary">
                                    No zones for this camera yet.
                                </Typography>
                            ) : (
                                <List dense>
                                    {zones.map((zone) => {
                                        const live = liveZoneStats[zone.zone_id] || { count: 0, objects: [] };
                                        return (
                                            <ListItem
                                                button
                                                key={zone.zone_id}
                                                selected={zone.zone_id === activeZoneId}
                                                onClick={() => handleZoneSelect(zone.zone_id)}
                                                sx={{ borderRadius: 2, mb: 1, cursor: 'pointer' }}
                                            >
                                                <Avatar sx={{ bgcolor: zone.color, width: 32, height: 32, fontSize: 14, mr: 1 }}>
                                                    {zone.name.slice(0, 2).toUpperCase()}
                                                </Avatar>
                                                <ListItemText
                                                    primary={zone.name}
                                                    secondary={`Mode: ${zone.attribution_mode} • ${live.count} inside`}
                                                />
                                                <Tooltip title="Delete zone">
                                                    <IconButton
                                                        size="small"
                                                        onClick={(event) => {
                                                            event.stopPropagation();
                                                            handleZoneDelete(zone.zone_id);
                                                        }}
                                                    >
                                                        <Trash size={16} />
                                                    </IconButton>
                                                </Tooltip>
                                            </ListItem>
                                        );
                                    })}
                                </List>
                            )}
                            {activeZoneStats && (
                                <Box sx={{ mt: 2, p: 2, borderRadius: 2, backgroundColor: alpha('#0ea5e9', 0.08) }}>
                                    <Typography variant="subtitle2" sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <ChartLine size={16} />
                                        Historical Stats
                                    </Typography>
                                    <Grid container spacing={1}>
                                        <Grid item xs={4}>
                                            <Typography variant="caption" color="text.secondary">Entries</Typography>
                                            <Typography variant="h6">{activeZoneStats.entries_count}</Typography>
                                        </Grid>
                                        <Grid item xs={4}>
                                            <Typography variant="caption" color="text.secondary">Avg Dwell (s)</Typography>
                                            <Typography variant="h6">{Math.round(activeZoneStats.average_dwell_seconds || 0)}</Typography>
                                        </Grid>
                                        <Grid item xs={4}>
                                            <Typography variant="caption" color="text.secondary">Presence (s)</Typography>
                                            <Typography variant="h6">{Math.round(activeZoneStats.total_presence_seconds || 0)}</Typography>
                                        </Grid>
                                    </Grid>
                                    <Typography variant="caption" color="text.secondary">
                                        Live count: {activeZoneStats.live_count}
                                    </Typography>
                                </Box>
                            )}
                            {activeLive && (
                                <Box sx={{ mt: 2 }}>
                                    <Typography variant="subtitle2" sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <UsersThree size={16} />
                                        Live Presence ({activeLive.count})
                                    </Typography>
                                    {activeLive.objects.length === 0 ? (
                                        <Typography variant="body2" color="text.secondary">
                                            No tracked objects inside right now.
                                        </Typography>
                                    ) : (
                                        <Stack spacing={1}>
                                            {activeLive.objects.map((obj) => (
                                                <Box key={obj.event_id} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 1, borderRadius: 1, backgroundColor: alpha('#ffffff', isDark ? 0.04 : 0.6) }}>
                                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                                        {obj.object_id}
                                                    </Typography>
                                                    <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                                        <TimerIcon size={14} />
                                                        {Math.round(obj.dwell_seconds)}s
                                                    </Typography>
                                                </Box>
                                            ))}
                                        </Stack>
                                    )}
                                </Box>
                            )}
                        </Paper>
                    </Grid>
                    <Grid item xs={12} md={8}>
                        <Grid container spacing={3} sx={{ minHeight: '600px' }}>
                            {cameras.length === 0 ? (
                                <Grid item xs={12}>
                                    <Paper sx={{
                                        p: 8,
                                        textAlign: 'center',
                                        background: isDark
                                            ? alpha(theme.palette.background.paper, 0.03)
                                            : 'rgba(255, 255, 255, 0.7)',
                                        backdropFilter: 'blur(20px)',
                                        border: `1px solid ${alpha(theme.palette.divider, isDark ? 0.08 : 0.12)}`,
                                        borderRadius: 3,
                                    }}>
                                        <Camera size={64} color="#06b6d4" weight="duotone" />
                                        <Typography variant="h5" sx={{ mt: 3, mb: 2, fontWeight: 600 }}>
                                            No Cameras Found
                                        </Typography>
                                        <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
                                            Add cameras in the settings to start live analysis
                                        </Typography>
                                        <Button
                                            variant="contained"
                                            onClick={() => window.location.href = '/settings'}
                                            startIcon={<Camera size={20} />}
                                            sx={{
                                                background: isDark
                                                    ? 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)'
                                                    : 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
                                                borderRadius: 2,
                                                px: 4,
                                                py: 1.5,
                                                textTransform: 'none',
                                                fontWeight: 600,
                                            }}
                                        >
                                            Go to Settings
                                        </Button>
                                    </Paper>
                                </Grid>
                            ) : (
                            cameras.map((camera) => (
                                <Grid item xs={12} key={camera.id}>
                                    <VideoContainer sx={{ height: '400px' }}>
                                        {camera.isStreaming ? (
                                            <>
                                                <Box 
                                                    ref={(node) => {
                                                        if (node) {
                                                            videoRefs.current[camera.id] = node;
                                                        }
                                                    }}
                                                    sx={{
                                                        position: 'relative',
                                                        width: '100%',
                                                        height: '100%',
                                                        borderRadius: 2,
                                                        overflow: 'hidden',
                                                }}>
                                                    <motion.img
                                                        src={camera.videoSrc || 'path/to/default/image.jpg'}
                                                        alt={`Stream from ${camera.source_name}`}
                                                        style={{ 
                                                            width: '100%',
                                                            height: '100%',
                                                            objectFit: 'cover'
                                                        }}
                                                        onLoad={(event) => handleFrameLoad(camera.id, event)}
                                                        initial={{ opacity: 0 }}
                                                        animate={{ opacity: 1 }}
                                                        transition={{ duration: 0.3 }}
                                                    />
                                                    <Box
                                                        onClick={(event) => handleOverlayClick(event, camera.id)}
                                                        sx={{
                                                            position: 'absolute',
                                                            inset: 0,
                                                            pointerEvents: drawingMode && selectedCameraId === camera.id ? 'auto' : 'none',
                                                            cursor: drawingMode && selectedCameraId === camera.id ? 'crosshair' : 'default',
                                                        }}
                                                    >
                                                        {drawingMode && selectedCameraId === camera.id && draftPolygon.length > 0 && (
                                                            <svg width="100%" height="100%">
                                                                <polyline
                                                                    points={getDraftPolylinePoints(camera.id)}
                                                                    fill="rgba(14,165,233,0.2)"
                                                                    stroke="#0ea5e9"
                                                                    strokeWidth="2"
                                                                />
                                                                {draftPolygon.map((_, index) => {
                                                                    const node = videoRefs.current[camera.id];
                                                                    const frame = frameMeta[camera.id];
                                                                    if (!node || !frame) return null;
                                                                    const rect = node.getBoundingClientRect();
                                                                    const [px, py] = draftPolygon[index];
                                                                    const cx = (px / frame.width) * rect.width;
                                                                    const cy = (py / frame.height) * rect.height;
                                                                    return (
                                                                        <circle key={index} cx={cx} cy={cy} r={4} fill="#0ea5e9" />
                                                                    );
                                                                })}
                                                            </svg>
                                                        )}
                                                    </Box>
                                                    
                                                    {/* Camera info overlay - top */}
                                                    <Box sx={{
                                                        position: 'absolute',
                                                        top: 0,
                                                        left: 0,
                                                        right: 0,
                                                        padding: 2,
                                                        background: 'linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, transparent 100%)',
                                                        display: 'flex',
                                                        justifyContent: 'space-between',
                                                        alignItems: 'center',
                                                    }}>
                                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                            <Camera size={20} color="#06b6d4" weight="duotone" />
                                                            <Typography sx={{
                                                                color: 'white',
                                                                fontSize: '1rem',
                                                                fontWeight: 600,
                                                                textTransform: 'capitalize'
                                                            }}>
                                                                {camera.source_name || `Camera ${camera.id}`}
                                                            </Typography>
                                                        </Box>
                                                        <Chip
                                                            label={modelInfo[selectedModel].title}
                                                            size="small"
                                                            sx={{
                                                                backgroundColor: alpha(modelInfo[selectedModel].color, 0.2),
                                                                color: 'white',
                                                                border: `1px solid ${alpha(modelInfo[selectedModel].color, 0.5)}`,
                                                                fontWeight: 500,
                                                            }}
                                                        />
                                                    </Box>
                                                    
                                                    {/* Bottom left corner stop button */}
                                                    <Box sx={{
                                                        position: 'absolute',
                                                        bottom: 16,
                                                        left: 16,
                                                        transition: 'opacity 0.3s',
                                                        opacity: 0.7,
                                                        '&:hover': {
                                                            opacity: 1
                                                        }
                                                    }}>
                                                        <IconButton
                                                            onClick={() => stopSingleCamera(camera.id)}
                                                            size="small"
                                                            sx={{
                                                                backgroundColor: 'rgba(239, 68, 68, 0.9)',
                                                                backdropFilter: 'blur(8px)',
                                                                color: 'white',
                                                                '&:hover': {
                                                                    backgroundColor: 'rgba(220, 38, 38, 0.95)',
                                                                    transform: 'scale(1.1)'
                                                                },
                                                                transition: 'all 0.2s ease',
                                                            }}
                                                        >
                                                            <Pause size={16} weight="bold" />
                                                        </IconButton>
                                                    </Box>
                                                </Box>
                                            </>
                                        ) : (
                                            <Box sx={{ 
                                                display: 'flex', 
                                                flexDirection: 'column', 
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                height: '100%',
                                                gap: 3,
                                                p: 4,
                                                textAlign: 'center'
                                            }}>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                                                    <Camera size={40} color="#06b6d4" weight="duotone" />
                                                    <Typography variant="h5" sx={{ 
                                                        color: isDark ? 'rgba(255, 255, 255, 0.9)' : theme.palette.text.primary,
                                                        fontWeight: 700,
                                                        textTransform: 'capitalize',
                                                    }}>
                                                        {camera.source_name || `Camera ${camera.id}`}
                                                    </Typography>
                                                </Box>
                                                <ModernButton
                                                    onClick={() => startSingleCamera(camera.id)}
                                                    startIcon={<Play size={20} weight="bold" />}
                                                >
                                                    Start Stream
                                                </ModernButton>
                                                <Typography variant="body2" color="text.secondary" sx={{ 
                                                    maxWidth: '80%',
                                                    mt: 2,
                                                    opacity: 0.8
                                                }}>
                                                    Click to begin real-time analysis
                                                </Typography>
                                            </Box>
                                        )}
                                    </VideoContainer>
                                </Grid>
                            )))}
                        </Grid>
                    </Grid>
                </Grid>
            </motion.div>
        </StyledContainer>
    );
};

export default LiveAnalysis;