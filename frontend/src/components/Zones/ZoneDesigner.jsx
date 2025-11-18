import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import {
  Box,
  Button,
  Stack,
  TextField,
  Typography,
  IconButton,
  alpha,
  Alert,
} from '@mui/material';
import { PlayArrow, Stop, Gesture, Backspace } from '@mui/icons-material';

const colorPalette = ['#06b6d4', '#a855f7', '#f97316', '#22c55e', '#ec4899'];

const ZoneDesigner = ({ cameraId, apiBase, onZoneSaved }) => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [frameUrl, setFrameUrl] = useState('');
  const [frameMeta, setFrameMeta] = useState(null);
  const [draftPolygon, setDraftPolygon] = useState([]);
  const [zoneName, setZoneName] = useState('');
  const [drawingMode, setDrawingMode] = useState(false);
  const [error, setError] = useState('');
  const frameRef = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (frameUrl) {
        URL.revokeObjectURL(frameUrl);
      }
    };
  }, [frameUrl]);

  useEffect(() => {
    if (!isStreaming || !cameraId) {
      return undefined;
    }
    
    let consecutiveErrors = 0;
    const maxErrors = 3;
    
    intervalRef.current = setInterval(async () => {
      try {
        const response = await axios.get(`${apiBase}/process_frame/${cameraId}`, { 
          responseType: 'blob',
          timeout: 5000,
        });
        
        if (response.status === 200 && response.data.size > 0) {
          const blobUrl = URL.createObjectURL(response.data);
          setFrameUrl((prev) => {
            if (prev) URL.revokeObjectURL(prev);
            return blobUrl;
          });
          setError(''); // Clear error on success
          consecutiveErrors = 0;
        } else {
          consecutiveErrors++;
          if (consecutiveErrors >= maxErrors) {
            setError('Camera stream not available. Please ensure the camera is started and streaming.');
          }
        }
      } catch (err) {
        consecutiveErrors++;
        if (err.response?.status === 404) {
          setError('Camera stream not started. Click "Start" to begin streaming.');
        } else if (err.code === 'ECONNABORTED') {
          setError('Request timeout. Camera may be busy or disconnected.');
        } else if (consecutiveErrors >= maxErrors) {
          setError(`Unable to fetch video frame: ${err.message || 'Unknown error'}`);
        }
      }
    }, 150);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isStreaming, cameraId, apiBase]);

  const startStream = async () => {
    if (!cameraId) {
      setError('No camera selected');
      return;
    }
    try {
      setError('Starting camera stream...');
      const response = await axios.post(`${apiBase}/start_camera_stream/${cameraId}`, {
        model_type: 'objectDetection',
      }, {
        timeout: 15000,
      });
      
      if (response.status === 200) {
        setIsStreaming(true);
        setError('Initializing stream, please wait...');
        // Wait longer for the stream to initialize and start capturing frames
        // This gives time for the camera thread to start and capture first frame
        setTimeout(() => {
          setError('');
        }, 3000);
      }
    } catch (err) {
      console.error('Error starting stream:', err);
      let errorMsg = 'Unable to start camera stream';
      
      if (err.response) {
        errorMsg = err.response.data?.detail || `Server error: ${err.response.status}`;
      } else if (err.request) {
        errorMsg = 'No response from server. Check if backend is running.';
      } else if (err.code === 'ECONNABORTED') {
        errorMsg = 'Request timeout. Camera may be busy or the stream URL may be invalid.';
      }
      
      setError(errorMsg);
      setIsStreaming(false);
    }
  };

  const stopStream = async () => {
    if (!cameraId) return;
    try {
      await axios.post(`${apiBase}/stop_camera_stream/${cameraId}`);
      setIsStreaming(false);
      setFrameUrl('');
      setDraftPolygon([]);
      setDrawingMode(false);
    } catch (err) {
      setError('Unable to stop camera stream');
    }
  };

  const handleOverlayClick = (event) => {
    if (!drawingMode || !cameraId) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    if (!frameMeta) return;
    const relX = event.clientX - bounds.left;
    const relY = event.clientY - bounds.top;
    const x = Number(((relX / bounds.width) * frameMeta.width).toFixed(2));
    const y = Number(((relY / bounds.height) * frameMeta.height).toFixed(2));
    setDraftPolygon((prev) => [...prev, [x, y]]);
  };

  const handleImageLoad = (event) => {
    const { naturalWidth, naturalHeight } = event.target;
    if (naturalWidth && naturalHeight) {
      setFrameMeta({ width: naturalWidth, height: naturalHeight });
    }
  };

  const getDraftPolylinePoints = () => {
    if (!frameRef.current || !frameMeta || draftPolygon.length === 0) {
      return '';
    }
    const rect = frameRef.current.getBoundingClientRect();
    return draftPolygon
      .map(([x, y]) => {
        const px = (x / frameMeta.width) * rect.width;
        const py = (y / frameMeta.height) * rect.height;
        return `${px},${py}`;
      })
      .join(' ');
  };

  const saveZone = async () => {
    if (!cameraId || draftPolygon.length < 3) {
      setError('At least 3 points are required to form a zone.');
      return;
    }
    try {
      const payload = {
        camera_id: cameraId,
        name: zoneName || `Zone ${Date.now()}`,
        polygon: draftPolygon,
        color: colorPalette[Math.floor(Math.random() * colorPalette.length)],
        attribution_mode: 'multiple',
        properties: { created_from: 'dwell-designer' },
      };
      await axios.post(`${apiBase}/zones`, payload);
      setDraftPolygon([]);
      setZoneName('');
      setDrawingMode(false);
      setError('');
      if (onZoneSaved) {
        onZoneSaved();
      }
    } catch (err) {
      setError('Failed to save zone');
    }
  };

  return (
    <Box sx={{ borderRadius: 3, p: 3, border: (theme) => `1px solid ${alpha(theme.palette.divider, 0.1)}` }}>
      <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 700 }}>
          Zone Designer
        </Typography>
        <Button
          startIcon={<PlayArrow />}
          variant="contained"
          disabled={!cameraId || isStreaming}
          onClick={startStream}
        >
          Start
        </Button>
        <Button
          startIcon={<Stop />}
          color="secondary"
          variant="outlined"
          disabled={!isStreaming}
          onClick={stopStream}
        >
          Stop
        </Button>
        <Button
          startIcon={<Gesture />}
          variant="text"
          disabled={!isStreaming}
          onClick={() => {
            setDrawingMode(true);
            setDraftPolygon([]);
            setError('');
          }}
        >
          Draw
        </Button>
        <IconButton
          onClick={() => setDraftPolygon((prev) => prev.slice(0, -1))}
          disabled={draftPolygon.length === 0}
        >
          <Backspace />
        </IconButton>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box
        ref={frameRef}
        sx={{
          position: 'relative',
          width: '100%',
          borderRadius: 3,
          overflow: 'hidden',
          background: (theme) => alpha(theme.palette.background.paper, 0.4),
          minHeight: 360,
        }}
      >
        {frameUrl ? (
          <img
            src={frameUrl}
            alt="Camera stream"
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            onLoad={handleImageLoad}
          />
        ) : (
          <Box
            sx={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'text.secondary',
              flexDirection: 'column',
              gap: 1,
            }}
          >
            <Typography variant="body1">
              {cameraId ? 'Start the camera stream to begin drawing zones.' : 'Select a camera to design zones.'}
            </Typography>
          </Box>
        )}

        <Box
          onClick={handleOverlayClick}
          sx={{
            position: 'absolute',
            inset: 0,
            cursor: drawingMode ? 'crosshair' : 'default',
            pointerEvents: drawingMode ? 'auto' : 'none',
          }}
        >
          {draftPolygon.length > 0 && (
            <svg width="100%" height="100%">
              <polyline
                points={getDraftPolylinePoints()}
                fill="rgba(6,182,212,0.18)"
                stroke="#06b6d4"
                strokeWidth="2"
              />
            </svg>
          )}
        </Box>
      </Box>

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mt: 3 }}>
        <TextField
          label="Zone name"
          value={zoneName}
          onChange={(event) => setZoneName(event.target.value)}
          fullWidth
          size="small"
        />
        <Button
          variant="contained"
          onClick={saveZone}
          disabled={draftPolygon.length < 3}
        >
          Save Zone
        </Button>
      </Stack>
      {drawingMode && (
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          Click on the video frame to place vertices. Points: {draftPolygon.length}
        </Typography>
      )}
    </Box>
  );
};

export default ZoneDesigner;

