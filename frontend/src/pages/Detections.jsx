import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Chip,
  TextField,
  InputAdornment,
  IconButton,
  useTheme,
  alpha,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
} from '@mui/material';
import { MagnifyingGlass, ArrowClockwise, Camera, Calendar } from '@phosphor-icons/react';
import { format } from 'date-fns';
import axios from 'axios';

const Detections = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCamera, setSelectedCamera] = useState('all');
  const [cameras, setCameras] = useState([]);

  const fetchDetections = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:8000/api/detections', {
        params: {
          limit: 100,
          offset: 0
        },
        timeout: 10000 // 10 second timeout
      });
      setDetections(response.data || []);
    } catch (error) {
      console.error('Error fetching detections:', error);
      if (error.code === 'ECONNABORTED') {
        console.error('Request timeout - backend may not be running');
      } else if (error.response) {
        console.error('Backend error:', error.response.status, error.response.data);
      } else if (error.request) {
        console.error('No response from backend - check if server is running');
      }
      setDetections([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchCameras = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/cameras');
      setCameras(response.data || []);
    } catch (error) {
      console.error('Error fetching cameras:', error);
    }
  };

  useEffect(() => {
    fetchDetections();
    fetchCameras();
  }, []);

  const filteredDetections = detections.filter(detection => {
    const matchesSearch = detection.class_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         detection.camera_name?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCamera = selectedCamera === 'all' || detection.camera_id === selectedCamera;
    return matchesSearch && matchesCamera;
  });

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const getClassColor = (className) => {
    const colors = {
      'person': 'primary',
      'car': 'success',
      'truck': 'warning',
      'bicycle': 'info',
      'motorcycle': 'secondary',
      'default': 'default'
    };
    return colors[className?.toLowerCase()] || colors.default;
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4, position: 'relative', zIndex: 1 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" sx={{ mb: 2, fontWeight: 700 }}>
          Detection History
        </Typography>
        <Typography variant="body1" color="text.secondary">
          View and search through all detected objects from your cameras
        </Typography>
      </Box>

      {/* Filters */}
      <Paper 
        sx={{ 
          p: 3, 
          mb: 3,
          background: theme.palette.mode === 'dark' 
            ? alpha(theme.palette.background.paper, 0.05)
            : 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(10px)',
          border: 'none',
        }}
      >
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Search by object type or camera..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 2,
                  backgroundColor: isDark ? alpha('#fff', 0.02) : alpha('#fff', 0.8),
                  '& fieldset': {
                    borderColor: alpha(theme.palette.divider, 0.2),
                  },
                  '&:hover fieldset': {
                    borderColor: alpha('#06b6d4', 0.4),
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: '#06b6d4',
                  },
                },
              }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <MagnifyingGlass size={20} color="#06b6d4" />
                  </InputAdornment>
                ),
              }}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Camera</InputLabel>
              <Select
                value={selectedCamera}
                onChange={(e) => setSelectedCamera(e.target.value)}
                label="Camera"
                sx={{
                  borderRadius: 2,
                  backgroundColor: isDark ? alpha('#fff', 0.02) : alpha('#fff', 0.8),
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: alpha(theme.palette.divider, 0.2),
                  },
                  '&:hover .MuiOutlinedInput-notchedOutline': {
                    borderColor: alpha('#06b6d4', 0.4),
                  },
                  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                    borderColor: '#06b6d4',
                  },
                }}
              >
                <MenuItem value="all">All Cameras</MenuItem>
                {cameras.map((camera) => (
                  <MenuItem key={camera.id} value={camera.id}>
                    {camera.source_name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <IconButton
              onClick={fetchDetections}
              sx={{
                backgroundColor: alpha(theme.palette.primary.main, 0.1),
                '&:hover': {
                  backgroundColor: alpha(theme.palette.primary.main, 0.2),
                },
              }}
            >
              <ArrowClockwise />
            </IconButton>
          </Grid>
        </Grid>
      </Paper>

      {/* Detections Table */}
      <TableContainer 
        component={Paper}
        sx={{
          background: theme.palette.mode === 'dark' 
            ? alpha(theme.palette.background.paper, 0.05)
            : 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(10px)',
          border: 'none',
        }}
      >
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Time</TableCell>
                  <TableCell>Object Type</TableCell>
                  <TableCell>Camera</TableCell>
                  <TableCell>Model</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredDetections
                  .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                  .map((detection) => (
                    <TableRow
                      key={detection.id}
                      sx={{
                        '&:hover': {
                          backgroundColor: isDark ? alpha(theme.palette.primary.main, 0.05) : alpha(theme.palette.primary.main, 0.08),
                        },
                      }}
                    >
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Calendar size={16} />
                          {format(new Date(detection.timestamp), 'MMM dd, HH:mm:ss')}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={detection.class_name}
                          size="small"
                          sx={{
                            backgroundColor: (() => {
                              const colorName = getClassColor(detection.class_name);
                              const color = colorName === 'default' ? '#9e9e9e' : theme.palette[colorName]?.main || '#9e9e9e';
                              return alpha(color, 0.1);
                            })(),
                            color: (() => {
                              const colorName = getClassColor(detection.class_name);
                              const color = colorName === 'default' ? '#9e9e9e' : theme.palette[colorName]?.main || '#9e9e9e';
                              return color;
                            })(),
                            border: (() => {
                              const colorName = getClassColor(detection.class_name);
                              const color = colorName === 'default' ? '#9e9e9e' : theme.palette[colorName]?.main || '#9e9e9e';
                              return `1px solid ${alpha(color, 0.3)}`;
                            })(),
                            fontWeight: 500,
                          }}
                        />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Camera size={16} />
                          {detection.camera_name}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {detection.model_type}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
            <TablePagination
              rowsPerPageOptions={[10, 25, 50, 100]}
              component="div"
              count={filteredDetections.length}
              rowsPerPage={rowsPerPage}
              page={page}
              onPageChange={handleChangePage}
              onRowsPerPageChange={handleChangeRowsPerPage}
            />
          </>
        )}
      </TableContainer>
    </Container>
  );
};

export default Detections;