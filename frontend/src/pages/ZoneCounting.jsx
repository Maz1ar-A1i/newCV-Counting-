import React, { useEffect, useState } from 'react';
import axios from 'axios';
import {
  Box,
  Grid,
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Stack,
  Button,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  TextField,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import DownloadIcon from '@mui/icons-material/Download';
import ZoneDesigner from '../components/Zones/ZoneDesigner';

const API_BASE = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

const ZoneCounting = () => {
  const [cameras, setCameras] = useState([]);
  const [filters, setFilters] = useState({
    cameraId: '',
    zoneId: '',
    objectType: '',
  });
  const [zones, setZones] = useState([]);
  const [liveZones, setLiveZones] = useState([]);
  const [events, setEvents] = useState([]);

  const loadCameras = async () => {
    const { data } = await axios.get(`${API_BASE}/api/cameras`);
    setCameras(data);
    if (!filters.cameraId && data.length) {
      setFilters((prev) => ({ ...prev, cameraId: data[0].id }));
    }
  };

  const loadZones = async () => {
    if (!filters.cameraId) {
      setZones([]);
      return;
    }
    const { data } = await axios.get(`${API_BASE}/zones`, {
      params: { camera_id: filters.cameraId },
    });
    setZones(data);
  };

  const loadLiveZones = async () => {
    const params = {};
    if (filters.cameraId) params.camera_id = filters.cameraId;
    if (filters.zoneId) params.zone_id = filters.zoneId;
    if (filters.objectType) params.object_type = filters.objectType;
    const { data } = await axios.get(`${API_BASE}/zone-counters/live`, { params });
    setLiveZones(data);
  };

  const loadEvents = async () => {
    const params = { limit: 200 };
    if (filters.cameraId) params.camera_id = filters.cameraId;
    if (filters.zoneId) params.zone_id = filters.zoneId;
    if (filters.objectType) params.object_type = filters.objectType;
    const { data } = await axios.get(`${API_BASE}/zone-counters/events`, { params });
    setEvents(data);
  };

  useEffect(() => {
    loadCameras();
  }, []);

  useEffect(() => {
    loadZones();
  }, [filters.cameraId]);

  useEffect(() => {
    loadLiveZones();
    loadEvents();
    const interval = setInterval(() => {
      loadLiveZones();
    }, 5000);
    return () => clearInterval(interval);
  }, [filters]);

  const handleFilterChange = (field, value) => {
    const normalizedValue = field === 'cameraId' && value !== '' ? Number(value) : value;
    setFilters((prev) => ({ ...prev, [field]: normalizedValue }));
  };

  const handleExport = (format = 'csv') => {
    const params = new URLSearchParams();
    if (filters.cameraId) params.append('camera_id', filters.cameraId);
    if (filters.zoneId) params.append('zone_id', filters.zoneId);
    if (filters.objectType) params.append('object_type', filters.objectType);
    params.append('format', format);
    window.open(`${API_BASE}/zone-counters/export?${params.toString()}`, '_blank');
  };

  return (
    <Box sx={{ px: 4, pb: 6 }}>
      <Box sx={{ mb: 5 }}>
        <Typography variant="h3" sx={{ fontWeight: 800 }}>
          Zone-Based Counting
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Count any object class per zone, analyze entries/exits, and export history.
        </Typography>
      </Box>

      <Paper sx={{ p: 3, borderRadius: 3, mb: 3 }}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Camera</InputLabel>
              <Select
                label="Camera"
                value={filters.cameraId}
                onChange={(event) => handleFilterChange('cameraId', event.target.value)}
              >
                {cameras.map((camera) => (
                  <MenuItem key={camera.id} value={camera.id}>
                    {camera.source_name || `Camera ${camera.id}`}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Zone</InputLabel>
              <Select
                label="Zone"
                value={filters.zoneId}
                onChange={(event) => handleFilterChange('zoneId', event.target.value)}
              >
                <MenuItem value="">All Zones</MenuItem>
                {zones.map((zone) => (
                  <MenuItem key={zone.zone_id} value={zone.zone_id}>
                    {zone.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={3}>
            <TextField
              size="small"
              label="Object class"
              placeholder="person, vehicle, bottle..."
              value={filters.objectType}
              onChange={(event) => handleFilterChange('objectType', event.target.value)}
              fullWidth
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <Stack direction="row" spacing={1} justifyContent="flex-end">
              <Button startIcon={<DownloadIcon />} onClick={() => handleExport('csv')}>
                Export CSV
              </Button>
              <Button variant="outlined" startIcon={<DownloadIcon />} onClick={() => handleExport('json')}>
                JSON
              </Button>
            </Stack>
          </Grid>
        </Grid>
      </Paper>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Grid container spacing={2}>
            {liveZones.map((zoneItem) => (
              <Grid item xs={12} md={6} key={zoneItem.zone.zone_id}>
                <Box
                  sx={{
                    borderRadius: 3,
                    p: 2.5,
                    background: (theme) => alpha(theme.palette.primary.main, 0.04),
                    border: (theme) => `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                  }}
                >
                  <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                    {zoneItem.zone.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Total current: {zoneItem.live.count}
                  </Typography>
                  <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: 'wrap' }}>
                    {Object.entries(zoneItem.totals).map(([objectType, summary]) => (
                      <Chip
                        key={objectType}
                        label={`${objectType}: in ${summary.current} • in+ ${summary.entered} • out ${summary.exited}`}
                        size="small"
                      />
                    ))}
                    {Object.keys(zoneItem.totals).length === 0 && (
                      <Typography variant="caption" color="text.secondary">
                        No events yet for this zone.
                      </Typography>
                    )}
                  </Stack>
                </Box>
              </Grid>
            ))}
            {liveZones.length === 0 && (
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary">
                  No zones found. Draw a zone on the selected camera to start counting.
                </Typography>
              </Grid>
            )}
          </Grid>

          <Paper sx={{ mt: 3, p: 3, borderRadius: 3 }}>
            <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>
              Recent Zone Events
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Timestamp</TableCell>
                    <TableCell>Zone</TableCell>
                    <TableCell>Class</TableCell>
                    <TableCell>Event</TableCell>
                    <TableCell>Object ID</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {events.map((event) => {
                    const zone = zones.find((z) => z.zone_id === event.zone_id);
                    return (
                      <TableRow key={event.event_id}>
                        <TableCell>{new Date(event.timestamp).toLocaleString()}</TableCell>
                        <TableCell>{zone?.name || event.zone_id}</TableCell>
                        <TableCell>{event.object_type}</TableCell>
                        <TableCell>
                          <Chip
                            label={event.event_type}
                            size="small"
                            color={event.event_type === 'enter' ? 'success' : 'warning'}
                          />
                        </TableCell>
                        <TableCell>{event.object_id}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <ZoneDesigner
            cameraId={filters.cameraId ? Number(filters.cameraId) : null}
            apiBase={API_BASE}
            onZoneSaved={() => {
              loadZones();
              loadLiveZones();
            }}
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export default ZoneCounting;

