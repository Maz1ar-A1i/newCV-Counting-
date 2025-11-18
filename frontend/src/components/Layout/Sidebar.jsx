import React, { useState } from 'react';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  useTheme,
  Tooltip,
  alpha,
} from '@mui/material';
import {
  Home,
  Dashboard,
  Videocam,
  Science,
  Settings,
  Brightness4,
  Brightness7,
  ChevronLeft,
  ChevronRight,
  AccessTime,
  Map,
} from '@mui/icons-material';
import { useLocation, useNavigate } from 'react-router-dom';
import logo from '../../assets/nielsen-ai-logo.avif';

const drawerWidth = 240;
const collapsedWidth = 64;

const menuItems = [
  { text: 'Home', icon: <Home />, path: '/' },
  { text: 'Dashboard', icon: <Dashboard />, path: '/dashboard' },
  { text: 'Live Analysis', icon: <Videocam />, path: '/live-analysis' },
  { text: 'Detections', icon: <Science />, path: '/detections' },
  { text: 'Dwell Time', icon: <AccessTime />, path: '/dwell-time' },
  { text: 'Zone Counting', icon: <Map />, path: '/zone-counting' },
  { text: 'Settings', icon: <Settings />, path: '/settings' },
];

const Sidebar = ({ toggleTheme }) => {
  const theme = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);
  const isDark = theme.palette.mode === 'dark';

  const toggleCollapse = () => {
    setCollapsed(!collapsed);
  };

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: collapsed ? collapsedWidth : drawerWidth,
        flexShrink: 0,
        transition: 'width 0.3s ease',
        '& .MuiDrawer-paper': {
          width: collapsed ? collapsedWidth : drawerWidth,
          boxSizing: 'border-box',
          border: 'none',
          background: isDark
            ? 'linear-gradient(180deg, rgba(15, 15, 25, 0.7) 0%, rgba(20, 20, 35, 0.7) 100%)'
            : 'linear-gradient(180deg, rgba(255, 255, 255, 0.5) 0%, rgba(252, 252, 253, 0.5) 100%)',
          backdropFilter: 'blur(10px)',
          borderRight: `1px solid ${alpha(theme.palette.divider, isDark ? 0.05 : 0.08)}`,
          transition: 'all 0.3s ease',
          overflowX: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: isDark
              ? 'radial-gradient(circle at 50% 0%, rgba(6, 182, 212, 0.05) 0%, transparent 50%)'
              : 'radial-gradient(circle at 50% 0%, rgba(6, 182, 212, 0.06) 0%, transparent 50%)',
            pointerEvents: 'none',
          },
        },
      }}
    >
      <Box
        sx={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          zIndex: 1,
        }}
      >
        <Box
          sx={{
            p: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: collapsed ? 'center' : 'space-between',
            minHeight: '80px',
            borderBottom: `1px solid ${alpha(theme.palette.divider, isDark ? 0.05 : 0.08)}`,
            background: isDark
              ? alpha('#06b6d4', 0.02)
              : alpha('#06b6d4', 0.03),
          }}
        >
          {!collapsed && (
            <Box
              component="img"
              src={logo}
              alt="Logo"
              onClick={() => navigate('/')}
              sx={{
                width: '180px',
                height: 'auto',
                maxHeight: '65px',
                objectFit: 'contain',
                filter: theme.palette.mode === 'light' ? 'brightness(0.3) contrast(1.2)' : 'none',
                cursor: 'pointer',
                transition: 'transform 0.2s ease',
                '&:hover': {
                  transform: 'scale(1.05)',
                },
              }}
            />
          )}
          <IconButton
            onClick={toggleCollapse}
            sx={{
              ml: collapsed ? 0 : 1,
              color: theme.palette.text.secondary,
              background: alpha(theme.palette.primary.main, 0.05),
              '&:hover': {
                backgroundColor: alpha(theme.palette.primary.main, 0.1),
                color: theme.palette.primary.main,
              },
            }}
          >
            {collapsed ? <ChevronRight /> : <ChevronLeft />}
          </IconButton>
        </Box>

        <List sx={{ flexGrow: 1, px: 1, py: 2 }}>
          {menuItems.map(({ text, icon, path }) => {
            const isActive = location.pathname === path;
            return (
              <ListItem key={text} disablePadding sx={{ mb: 0.5 }}>
                <Tooltip title={collapsed ? text : ''} placement="right">
                  <ListItemButton
                    selected={isActive}
                    onClick={() => navigate(path)}
                    sx={{
                      borderRadius: 2.5,
                      justifyContent: collapsed ? 'center' : 'flex-start',
                      px: collapsed ? 1.5 : 2.5,
                      py: 1.25,
                      position: 'relative',
                      overflow: 'hidden',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        background: isActive
                          ? `linear-gradient(135deg, ${alpha('#06b6d4', 0.1)} 0%, ${alpha('#0891b2', 0.1)} 100%)`
                          : 'transparent',
                        transition: 'all 0.3s ease',
                      },
                      '&.Mui-selected': {
                        backgroundColor: 'transparent',
                        '&:hover': {
                          backgroundColor: 'transparent',
                          '&::before': {
                            background: `linear-gradient(135deg, ${alpha('#06b6d4', 0.2)} 0%, ${alpha('#0891b2', 0.2)} 100%)`,
                          },
                        },
                      },
                      '&:hover': {
                        backgroundColor: 'transparent',
                        '&::before': {
                          background: alpha(theme.palette.primary.main, 0.08),
                        },
                      },
                    }}
                  >
                    <ListItemIcon
                      sx={{
                        minWidth: collapsed ? 0 : 40,
                        color: isActive
                          ? '#06b6d4'
                          : theme.palette.text.secondary,
                        justifyContent: 'center',
                        position: 'relative',
                        zIndex: 1,
                        transition: 'all 0.3s ease',
                      }}
                    >
                      {icon}
                    </ListItemIcon>
                    {!collapsed && (
                      <ListItemText
                        primary={text}
                        sx={{
                          position: 'relative',
                          zIndex: 1,
                          '& .MuiListItemText-primary': {
                            fontWeight: isActive ? 600 : 400,
                            fontSize: '0.95rem',
                            color: isActive
                              ? '#06b6d4'
                              : theme.palette.text.secondary,
                            transition: 'all 0.3s ease',
                          },
                        }}
                      />
                    )}
                    {isActive && !collapsed && (
                      <Box
                        sx={{
                          position: 'absolute',
                          left: 0,
                          top: '50%',
                          transform: 'translateY(-50%)',
                          width: 4,
                          height: '70%',
                          background: 'linear-gradient(180deg, #06b6d4 0%, #0891b2 100%)',
                          borderRadius: '0 4px 4px 0',
                        }}
                      />
                    )}
                  </ListItemButton>
                </Tooltip>
              </ListItem>
            );
          })}
        </List>

        <Box sx={{ p: 2, borderTop: `1px solid ${alpha(theme.palette.divider, isDark ? 0.05 : 0.08)}` }}>
          <Tooltip title={collapsed ? (theme.palette.mode === 'dark' ? 'Light Mode' : 'Dark Mode') : ''} placement="right">
            <IconButton
              onClick={toggleTheme}
              sx={{
                width: '100%',
                borderRadius: 2.5,
                p: 1.5,
                color: theme.palette.text.secondary,
                background: alpha(theme.palette.action.hover, 0.04),
                '&:hover': {
                  background: alpha('#06b6d4', 0.08),
                  color: theme.palette.primary.main,
                },
              }}
            >
              {theme.palette.mode === 'dark' ? <Brightness7 /> : <Brightness4 />}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
    </Drawer>
  );
};

export default Sidebar;