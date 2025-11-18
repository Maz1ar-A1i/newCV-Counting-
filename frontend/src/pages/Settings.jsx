import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Card,
  CardContent,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Snackbar,
  Tooltip,
  InputAdornment,
  Container,
  Tab,
  Tabs,
  Avatar,
  alpha,
  useTheme
} from '@mui/material';
import { 
  Delete, 
  Edit, 
  Videocam, 
  Add, 
  LocationOn, 
  Info,
  Check,
  Close,
  Settings as SettingsIcon,
  CameraAlt,
  Security,
  Notifications,
  Storage,
  Circle
} from '@mui/icons-material';
import { Camera, Shield, Bell, Database, Plus, MapPin, Pencil, Trash } from '@phosphor-icons/react';
import { styled } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

// Styled components
const CameraCard = styled(Card)(({ theme }) => ({
  borderRadius: 12,
  background: theme.palette.mode === 'dark' 
    ? alpha(theme.palette.background.paper, 0.05)
    : 'rgba(255, 255, 255, 0.95)',
  backdropFilter: 'blur(10px)',
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  cursor: 'pointer',
  border: `1px solid ${alpha(theme.palette.divider, theme.palette.mode === 'dark' ? 0.08 : 0.12)}`,
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: theme.palette.mode === 'dark'
      ? '0 12px 24px rgba(6, 182, 212, 0.2)'
      : '0 12px 24px rgba(6, 182, 212, 0.15)',
    borderColor: alpha(theme.palette.primary.main, 0.3),
  },
}));

const TabPanel = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ py: 4 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const Settings = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  const [showCameraDialog, setShowCameraDialog] = useState(false);
  const [cameras, setCameras] = useState([]);
  const [editingCamera, setEditingCamera] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [cameraData, setCameraData] = useState({
    source_name: '',
    stream_type: 'live',
    stream: '',
    location: ''
  });

  useEffect(() => {
    fetchCameras();
  }, []);

  const fetchCameras = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/cameras');
      setCameras(response.data);
    } catch (error) {
      console.error('Error fetching cameras:', error);
      showSnackbar('Error fetching cameras', 'error');
    }
  };

  const handleOpenDialog = (camera = null) => {
    if (camera) {
      setEditingCamera(camera);
      setCameraData({
        source_name: camera.source_name,
        stream_type: camera.stream_type,
        stream: camera.stream,
        location: camera.location
      });
    } else {
      setEditingCamera(null);
      setCameraData({
        source_name: '',
        stream_type: 'live',
        stream: '',
        location: ''
      });
    }
    setShowCameraDialog(true);
  };

  const handleCloseDialog = () => {
    setShowCameraDialog(false);
    setEditingCamera(null);
    setCameraData({
      source_name: '',
      stream_type: 'live',
      stream: '',
      location: ''
    });
  };

  const showSnackbar = (message, severity = 'success') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      if (editingCamera) {
        await axios.put(`http://localhost:8000/api/cameras/${editingCamera.id}`, cameraData);
        showSnackbar('Camera updated successfully');
      } else {
        const response = await axios.post('http://localhost:8000/api/create_camera', cameraData, {
          timeout: 10000,
          validateStatus: (status) => status < 500 // Don't throw on 4xx errors
        });
        
        if (response.status === 200 || response.status === 201) {
          showSnackbar('Camera added successfully');
        } else {
          showSnackbar(response.data?.detail || 'Error adding camera', 'error');
          return; // Don't close dialog on error
        }
      }
      
      handleCloseDialog();
      fetchCameras();
      
    } catch (error) {
      console.error('Error saving camera:', error);
      let errorMessage = 'Error saving camera';
      
      if (error.response) {
        // Backend responded with error
        errorMessage = error.response.data?.detail || error.response.data?.message || `Server error: ${error.response.status}`;
      } else if (error.request) {
        // Request made but no response
        errorMessage = 'No response from server. Please check if the backend is running.';
      } else if (error.code === 'ECONNABORTED') {
        errorMessage = 'Request timeout. Please try again.';
      } else {
        errorMessage = error.message || 'Unknown error occurred';
      }
      
      showSnackbar(errorMessage, 'error');
    }
  };

  const handleDeleteCamera = async (cameraId) => {
    try {
      await axios.delete(`http://localhost:8000/api/cameras/${cameraId}`);
      fetchCameras();
      showSnackbar('Camera deleted successfully');
    } catch (error) {
      console.error('Error deleting camera:', error);
      showSnackbar('Error deleting camera', 'error');
    }
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  return (
    <Box>
      {/* Header */}
      <Container maxWidth="xl" sx={{ pt: 4, pb: 2 }}>
        <Typography 
          variant="h3" 
          gutterBottom 
          sx={{ 
            fontWeight: 800,
            background: isDark
              ? 'linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)'
              : 'linear-gradient(135deg, #0891b2 0%, #2563eb 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          Settings
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Manage your cameras and system preferences
        </Typography>
      </Container>

      {/* Tabs */}
      <Box sx={{ 
        borderBottom: `1px solid ${alpha(theme.palette.divider, 0.08)}`,
        background: isDark
          ? 'rgba(255, 255, 255, 0.01)'
          : 'rgba(255, 255, 255, 0.4)',
        backdropFilter: 'blur(20px)',
      }}>
        <Container maxWidth="xl">
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange}
            sx={{ 
              '& .MuiTab-root': {
                color: theme.palette.text.secondary,
                py: 3,
                fontWeight: 500,
                fontSize: '0.95rem',
                textTransform: 'none',
                transition: 'all 0.3s ease',
                '&.Mui-selected': {
                  color: '#06b6d4',
                  fontWeight: 600,
                },
                '&:hover': {
                  color: '#06b6d4',
                  backgroundColor: alpha('#06b6d4', 0.08),
                },
              },
              '& .MuiTabs-indicator': {
                backgroundColor: '#06b6d4',
                height: 3,
                borderRadius: '3px 3px 0 0',
              },
            }}
          >
            <Tab icon={<Camera size={20} />} label="Cameras" iconPosition="start" />
            <Tab icon={<Shield size={20} />} label="Security" iconPosition="start" />
            <Tab icon={<Bell size={20} />} label="Notifications" iconPosition="start" />
            <Tab icon={<Database size={20} />} label="Storage" iconPosition="start" />
          </Tabs>
        </Container>
      </Box>

      {/* Content */}
      <Container maxWidth="xl">
        <TabPanel value={tabValue} index={0}>
          {/* Camera Management Tab */}
          <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box>
              <Typography variant="h4" fontWeight={700} gutterBottom>
                Camera Management
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Add and manage your security cameras
              </Typography>
            </Box>
            <Button
              variant="contained"
              startIcon={<Plus />}
              onClick={() => handleOpenDialog()}
              sx={{ 
                borderRadius: 2,
                textTransform: 'none',
                px: 4,
                py: 1.5,
                background: isDark
                  ? 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)'
                  : 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
                boxShadow: isDark
                  ? '0 8px 32px rgba(6, 182, 212, 0.4)'
                  : '0 8px 32px rgba(14, 165, 233, 0.3)',
                '&:hover': {
                  background: isDark
                    ? 'linear-gradient(135deg, #0891b2 0%, #0e7490 100%)'
                    : 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
                  boxShadow: isDark
                    ? '0 12px 40px rgba(6, 182, 212, 0.5)'
                    : '0 12px 40px rgba(14, 165, 233, 0.4)',
                },
              }}
            >
              Add New Camera
            </Button>
          </Box>

          {/* Camera Grid */}
          <Grid container spacing={3}>
            {cameras.map((camera) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={camera.id}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <CameraCard>
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                          <Avatar sx={{ 
                            bgcolor: alpha('#06b6d4', 0.1), 
                            width: 44, 
                            height: 44,
                            border: `2px solid ${alpha('#06b6d4', 0.2)}`,
                          }}>
                            <Camera size={24} color="#06b6d4" weight="duotone" />
                          </Avatar>
                          <Box>
                            <Typography variant="h6" fontWeight={600}>
                              {camera.source_name}
                            </Typography>
                            <Chip 
                              icon={<Circle sx={{ fontSize: 8 }} />}
                              label={camera.stream_type} 
                              size="small" 
                              sx={{ 
                                mt: 0.5,
                                bgcolor: camera.stream_type === 'live' 
                                  ? alpha(theme.palette.success.main, 0.1)
                                  : alpha(theme.palette.info.main, 0.1),
                                color: camera.stream_type === 'live' 
                                  ? theme.palette.success.main
                                  : theme.palette.info.main,
                                border: `1px solid ${camera.stream_type === 'live' 
                                  ? alpha(theme.palette.success.main, 0.3)
                                  : alpha(theme.palette.info.main, 0.3)}`,
                                '& .MuiChip-icon': {
                                  color: 'inherit',
                                },
                              }}
                            />
                          </Box>
                        </Box>
                        <Box>
                          <Tooltip title="Edit">
                            <IconButton 
                              size="small" 
                              onClick={() => handleOpenDialog(camera)}
                              sx={{ 
                                color: theme.palette.primary.main,
                                '&:hover': {
                                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                                },
                              }}
                            >
                              <Pencil size={18} />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton 
                              size="small" 
                              onClick={() => handleDeleteCamera(camera.id)}
                              sx={{ 
                                color: theme.palette.error.main,
                                '&:hover': {
                                  bgcolor: alpha(theme.palette.error.main, 0.1),
                                },
                              }}
                            >
                              <Trash size={18} />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </Box>
                      
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <MapPin size={16} color={theme.palette.text.secondary} />
                          <Typography variant="body2" color="text.secondary">
                            {camera.location || 'No location set'}
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Info sx={{ fontSize: 16, color: 'text.secondary' }} />
                          <Typography variant="body2" color="text.secondary" noWrap>
                            {camera.stream}
                          </Typography>
                        </Box>
                      </Box>
                    </CardContent>
                  </CameraCard>
                </motion.div>
              </Grid>
            ))}
            
            {/* Add Camera Card */}
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: cameras.length * 0.1 }}
              >
                <CameraCard 
                  sx={{ 
                    border: `2px dashed ${alpha(theme.palette.primary.main, 0.3)}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: 180,
                    background: isDark
                      ? alpha(theme.palette.primary.main, 0.02)
                      : alpha(theme.palette.primary.main, 0.03),
                    '&:hover': {
                      borderColor: theme.palette.primary.main,
                      background: isDark
                        ? alpha(theme.palette.primary.main, 0.05)
                        : alpha(theme.palette.primary.main, 0.08),
                    }
                  }}
                  onClick={() => handleOpenDialog()}
                >
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Plus size={48} color={theme.palette.primary.main} weight="light" />
                    <Typography variant="body1" color="primary" sx={{ mt: 1, fontWeight: 500 }}>
                      Add New Camera
                    </Typography>
                  </CardContent>
                </CameraCard>
              </motion.div>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Box sx={{ maxWidth: 'md' }}>
            <Typography variant="h4" fontWeight={700} gutterBottom>
              Security Settings
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
              Configure security and access control settings
            </Typography>
            <Alert 
              severity="info" 
              sx={{ 
                bgcolor: isDark 
                  ? alpha(theme.palette.info.main, 0.1)
                  : alpha(theme.palette.info.main, 0.08),
                border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`,
              }}
            >
              Security settings will be available in a future update.
            </Alert>
          </Box>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Box sx={{ maxWidth: 'md' }}>
            <Typography variant="h4" fontWeight={700} gutterBottom>
              Notification Preferences
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
              Manage alerts and notification settings
            </Typography>
            <Alert 
              severity="info" 
              sx={{ 
                bgcolor: isDark 
                  ? alpha(theme.palette.info.main, 0.1)
                  : alpha(theme.palette.info.main, 0.08),
                border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`,
              }}
            >
              Notification settings will be available in a future update.
            </Alert>
          </Box>
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <Box sx={{ maxWidth: 'md' }}>
            <Typography variant="h4" fontWeight={700} gutterBottom>
              Storage Management
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
              Manage recording storage and retention
            </Typography>
            <Alert 
              severity="info" 
              sx={{ 
                bgcolor: isDark 
                  ? alpha(theme.palette.info.main, 0.1)
                  : alpha(theme.palette.info.main, 0.08),
                border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`,
              }}
            >
              Storage settings will be available in a future update.
            </Alert>
          </Box>
        </TabPanel>
      </Container>

      {/* Camera Dialog */}
      <Dialog 
        open={showCameraDialog} 
        onClose={handleCloseDialog}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: { 
            borderRadius: 3,
            background: isDark 
              ? alpha(theme.palette.background.paper, 0.9)
              : 'rgba(255, 255, 255, 0.98)',
            backdropFilter: 'blur(20px)',
          }
        }}
      >
        <DialogTitle sx={{ pb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Camera size={24} color="#06b6d4" weight="duotone" />
            <Typography variant="h6">
              {editingCamera ? 'Edit Camera' : 'Add New Camera'}
            </Typography>
          </Box>
        </DialogTitle>
        <form onSubmit={handleSubmit}>
          <DialogContent>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <TextField
                  autoFocus
                  fullWidth
                  label="Camera Name"
                  value={cameraData.source_name}
                  onChange={(e) => setCameraData({...cameraData, source_name: e.target.value})}
                  required
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Camera size={20} weight="duotone" />
                      </InputAdornment>
                    ),
                  }}
                />
              </Grid>
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel>Stream Type</InputLabel>
                  <Select
                    value={cameraData.stream_type}
                    onChange={(e) => setCameraData({...cameraData, stream_type: e.target.value})}
                    label="Stream Type"
                    required
                  >
                    <MenuItem value="live">Live Stream</MenuItem>
                    <MenuItem value="recorded">Recorded</MenuItem>
                    <MenuItem value="rtsp">RTSP</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Stream URL/Path"
                  value={cameraData.stream}
                  onChange={(e) => setCameraData({...cameraData, stream: e.target.value})}
                  required
                  placeholder={cameraData.stream_type === 'live' ? '0' : 'rtsp://...'}
                  helperText={cameraData.stream_type === 'live' ? 'Enter webcam index (0 for default)' : 'Enter stream URL'}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Location"
                  value={cameraData.location}
                  onChange={(e) => setCameraData({...cameraData, location: e.target.value})}
                  placeholder="e.g., Front Door, Parking Lot"
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <MapPin size={20} />
                      </InputAdornment>
                    ),
                  }}
                />
              </Grid>
            </Grid>
          </DialogContent>
          <DialogActions sx={{ px: 3, pb: 3 }}>
            <Button onClick={handleCloseDialog} startIcon={<Close />}>
              Cancel
            </Button>
            <Button 
              type="submit" 
              variant="contained" 
              startIcon={<Check />}
              sx={{ 
                background: isDark
                  ? 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)'
                  : 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
                boxShadow: '0 4px 20px rgba(6, 182, 212, 0.3)',
              }}
            >
              {editingCamera ? 'Update' : 'Add'} Camera
            </Button>
          </DialogActions>
        </form>
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => setSnackbar({ ...snackbar, open: false })} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Settings;