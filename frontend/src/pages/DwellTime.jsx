import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  Box,
  Grid,
  Paper,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Stack,
  Table,
  TableHead,
  TableBody,
  TableCell,
  TableRow,
  TableContainer,
  Avatar,
  Switch,
  FormControlLabel,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import ZoneDesigner from '../components/Zones/ZoneDesigner';

const API_BASE = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

const formatDuration = (seconds = 0) => {
  const total = Math.max(0, Math.floor(seconds));
  const hrs = String(Math.floor(total / 3600)).padStart(2, '0');
  const mins = String(Math.floor((total % 3600) / 60)).padStart(2, '0');
  const secs = String(total % 60).padStart(2, '0');
  return `${hrs}:${mins}:${secs}`;
};

const DwellTime = () => {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [zones, setZones] = useState([]);
  const [targets, setTargets] = useState([]);
  const [liveSessions, setLiveSessions] = useState([]);
  const [sessionHistory, setSessionHistory] = useState([]);
  const [selectedTarget, setSelectedTarget] = useState('');
  const [formState, setFormState] = useState({
    name: '',
    zoneIds: [],
    matchThreshold: 0.45,
    file: null,
  });
  const [loading, setLoading] = useState(false);

  const loadCameras = async () => {
    const { data } = await axios.get(`${API_BASE}/api/cameras`);
    setCameras(data);
    if (!selectedCamera && data.length) {
      setSelectedCamera(data[0].id);
    }
  };

  const loadZones = async () => {
    if (!selectedCamera) return;
    const { data } = await axios.get(`${API_BASE}/zones`, {
      params: { camera_id: selectedCamera },
    });
    setZones(data);
  };

  const loadTargets = async () => {
    const { data } = await axios.get(`${API_BASE}/dwell/targets`);
    setTargets(data);
  };

  const loadLiveSessions = async () => {
    const { data } = await axios.get(`${API_BASE}/dwell/live`);
    setLiveSessions(data.sessions || []);
  };

  const loadSessionHistory = async () => {
    const params = { limit: 100 };
    if (selectedTarget) {
      params.target_id = selectedTarget;
    }
    const { data } = await axios.get(`${API_BASE}/dwell/sessions`, { params });
    setSessionHistory(data);
  };

  useEffect(() => {
    loadCameras();
    loadTargets();
  }, []);

  useEffect(() => {
    if (selectedCamera) {
      loadZones();
    } else {
      setZones([]);
    }
  }, [selectedCamera]);

  useEffect(() => {
    loadSessionHistory();
  }, [selectedTarget]);

  useEffect(() => {
    loadLiveSessions();
    const interval = setInterval(loadLiveSessions, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleFormChange = (field, value) => {
    setFormState((prev) => ({ ...prev, [field]: value }));
  };

  const handleCreateTarget = async () => {
    if (!formState.name || !formState.file || formState.zoneIds.length === 0 || !selectedCamera) {
      alert('Please fill in all required fields: name, face image, and at least one zone');
      return;
    }
    const formData = new FormData();
    formData.append('name', formState.name);
    formData.append('camera_id', selectedCamera);
    formData.append('zone_ids', JSON.stringify(formState.zoneIds));
    formData.append('match_threshold', formState.matchThreshold);
    formData.append('face_image', formState.file);
    try {
      setLoading(true);
      const response = await axios.post(`${API_BASE}/dwell/targets`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000, // 30 second timeout for face recognition processing
      });
      
      if (response.status === 201 || response.status === 200) {
        setFormState({
          name: '',
          zoneIds: [],
          matchThreshold: 0.45,
          file: null,
        });
        loadTargets();
        loadSessionHistory();
        alert('Target enrolled successfully!');
      }
    } catch (error) {
      console.error('Error creating target:', error);
      let errorMessage = 'Failed to enroll target';
      
      if (error.response) {
        errorMessage = error.response.data?.detail || error.response.data?.message || `Server error: ${error.response.status}`;
      } else if (error.request) {
        errorMessage = 'No response from server. Please check if the backend is running and face_recognition library is installed.';
      } else if (error.code === 'ECONNABORTED') {
        errorMessage = 'Request timeout. Face recognition processing may take longer.';
      } else {
        errorMessage = error.message || 'Unknown error occurred';
      }
      
      alert(`Error: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleTarget = async (target) => {
    await axios.put(`${API_BASE}/dwell/targets/${target.target_id}`, {
      is_active: !target.is_active,
    });
    loadTargets();
  };

  const handleDeleteTarget = async (target) => {
    await axios.delete(`${API_BASE}/dwell/targets/${target.target_id}`);
    loadTargets();
    loadSessionHistory();
  };

  const targetOptions = useMemo(
    () => targets.map((target) => ({ label: target.name, value: target.target_id })),
    [targets]
  );

  return (
    <Box sx={{ px: 4, pb: 6 }}>
      <Box sx={{ mb: 5 }}>
        <Typography variant="h3" sx={{ fontWeight: 800 }}>
          Dwell Time Analysis
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Configure a target face, draw dedicated zones, and monitor dwell duration in real time.
        </Typography>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper
            sx={{
              p: 3,
              borderRadius: 3,
              background: (theme) => alpha(theme.palette.background.paper, 0.7),
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>
              Target Enrollment
            </Typography>
            <Stack spacing={2}>
              <TextField
                label="Target name"
                value={formState.name}
                onChange={(event) => handleFormChange('name', event.target.value)}
                size="small"
              />
              <FormControl fullWidth size="small">
                <InputLabel>Camera</InputLabel>
                <Select
                  label="Camera"
                  value={selectedCamera ?? ''}
                  onChange={(event) => setSelectedCamera(event.target.value || null)}
                >
                  {cameras.map((camera) => (
                    <MenuItem key={camera.id} value={camera.id}>
                      {camera.source_name || `Camera ${camera.id}`}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl fullWidth size="small">
                <InputLabel>Zones</InputLabel>
                <Select
                  multiple
                  label="Zones"
                  value={formState.zoneIds}
                  onChange={(event) => handleFormChange('zoneIds', event.target.value)}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => {
                        const zone = zones.find((z) => z.zone_id === value);
                        return (
                          <Chip key={value} label={zone?.name || value} />
                        );
                      })}
                    </Box>
                  )}
                >
                  {zones.map((zone) => (
                    <MenuItem key={zone.zone_id} value={zone.zone_id}>
                      {zone.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <TextField
                label="Match threshold"
                type="number"
                size="small"
                inputProps={{ step: 0.01, min: 0.3, max: 0.6 }}
                value={formState.matchThreshold}
                onChange={(event) => handleFormChange('matchThreshold', Number(event.target.value))}
              />
              <Button
                variant="outlined"
                component="label"
                fullWidth
              >
                {formState.file ? formState.file.name : 'Upload Face Image'}
                <input
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={(event) => handleFormChange('file', event.target.files?.[0] || null)}
                />
              </Button>
              <Button
                variant="contained"
                onClick={handleCreateTarget}
                disabled={
                  loading ||
                  !formState.name ||
                  !formState.file ||
                  formState.zoneIds.length === 0 ||
                  !selectedCamera
                }
              >
                Save Target
              </Button>
            </Stack>
          </Paper>

          <Paper sx={{ mt: 3, p: 3, borderRadius: 3 }}>
            <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>
              Active Targets
            </Typography>
            <Stack spacing={2}>
              {targets.map((target) => (
                <Box
                  key={target.target_id}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    p: 1.5,
                    borderRadius: 2,
                    border: (theme) => `1px solid ${alpha(theme.palette.divider, 0.12)}`,
                  }}
                >
                  <Box>
                    <Typography sx={{ fontWeight: 600 }}>{target.name}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Zones: {target.zone_ids.length}
                    </Typography>
                  </Box>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <FormControlLabel
                      control={
                        <Switch
                          checked={target.is_active}
                          onChange={() => handleToggleTarget(target)}
                          size="small"
                        />
                      }
                      label={target.is_active ? 'On' : 'Off'}
                    />
                    <Button color="error" size="small" onClick={() => handleDeleteTarget(target)}>
                      Remove
                    </Button>
                  </Stack>
                </Box>
              ))}
              {targets.length === 0 && (
                <Typography variant="body2" color="text.secondary">
                  No targets enrolled yet.
                </Typography>
              )}
            </Stack>
          </Paper>
        </Grid>

        <Grid item xs={12} md={8}>
          <ZoneDesigner cameraId={selectedCamera} apiBase={API_BASE} onZoneSaved={loadZones} />

          <Paper sx={{ mt: 3, p: 3, borderRadius: 3 }}>
            <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} justifyContent="space-between" sx={{ mb: 2 }}>
              <Typography variant="h6" sx={{ fontWeight: 700 }}>
                Live Dwell Sessions
              </Typography>
              <FormControl size="small" sx={{ minWidth: 220 }}>
                <InputLabel>Filter by Target</InputLabel>
                <Select
                  label="Filter by Target"
                  value={selectedTarget}
                  onChange={(event) => setSelectedTarget(event.target.value)}
                >
                  <MenuItem value="">All Targets</MenuItem>
                  {targetOptions.map((target) => (
                    <MenuItem key={target.value} value={target.value}>
                      {target.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Stack>
            <Grid container spacing={2}>
              {liveSessions.length === 0 && (
                <Grid item xs={12}>
                  <Typography variant="body2" color="text.secondary">
                    No active sessions right now.
                  </Typography>
                </Grid>
              )}
              {liveSessions.map((session) => (
                <Grid item xs={12} md={6} key={session.session_id || `${session.target_id}-${session.zone_id}`}>
                  <Box
                    sx={{
                      borderRadius: 3,
                      p: 2,
                      background: (theme) => alpha(theme.palette.success.main, 0.08),
                    }}
                  >
                    <Stack direction="row" spacing={2} alignItems="center">
                      <Avatar>{session.target_name?.slice(0, 2) || 'DT'}</Avatar>
                      <Box>
                        <Typography sx={{ fontWeight: 600 }}>
                          {session.target_name || session.target_id}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Zone: {session.zone_name || session.zone_id}
                        </Typography>
                      </Box>
                    </Stack>
                    <Stack direction="row" spacing={3} sx={{ mt: 2 }}>
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          Arrival
                        </Typography>
                        <Typography>{new Date(session.entry_ts).toLocaleTimeString()}</Typography>
                      </Box>
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          Live dwell
                        </Typography>
                        <Typography sx={{ fontWeight: 700 }}>
                          {formatDuration(session.dwell_seconds)}
                        </Typography>
                      </Box>
                    </Stack>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Paper>

          <Paper sx={{ mt: 3, p: 3, borderRadius: 3 }}>
            <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>
              Session History
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Target</TableCell>
                    <TableCell>Zone</TableCell>
                    <TableCell>Arrival</TableCell>
                    <TableCell>Exit</TableCell>
                    <TableCell align="right">Dwell</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {sessionHistory.map((session) => {
                    const target = targets.find((t) => t.target_id === session.target_id);
                    const zone = zones.find((z) => z.zone_id === session.zone_id);
                    return (
                      <TableRow key={session.session_id}>
                        <TableCell>{target?.name || session.target_id}</TableCell>
                        <TableCell>{zone?.name || session.zone_id}</TableCell>
                        <TableCell>{new Date(session.entry_ts).toLocaleString()}</TableCell>
                        <TableCell>{session.exit_ts ? new Date(session.exit_ts).toLocaleString() : '—'}</TableCell>
                        <TableCell align="right">
                          {session.dwell_seconds ? formatDuration(session.dwell_seconds) : '—'}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DwellTime;

