
<img width="1704" height="898" alt="Screenshot 2025-07-31 at 13 29 08" src="https://github.com/user-attachments/assets/543be458-81d6-45d6-82cb-fd751d31135f" />
<img width="1660" height="894" alt="Screenshot 2025-07-31 at 13 29 32" src="https://github.com/user-attachments/assets/891e0f84-02d7-4822-ac8a-282f8a2805d5" />


# AI Vision System

A real-time AI vision processing system with camera management, object detection, and monitoring capabilities.

## Features

- **Real-time Analysis** - Process video streams with AI models (object detection, segmentation, pose estimation)
- **Camera Management** - Monitor and control multiple camera streams
- **Dwell Time Module** - Upload a face sample, draw custom zones, and measure dwell duration with history logs
- **Zone-Based Counting** - Count any YOLO class per zone with entry/exit analytics and CSV/JSON exports
- **Modern UI** - React + Vite frontend with Material-UI
- **Fast API Backend** - RESTful endpoints and real-time processing
- **PostgreSQL Database** - Reliable data storage
- **Docker Support** - Easy deployment with docker-compose

## Prerequisites

- **Node.js** (v14+ recommended)
- **Python** (v3.10+ recommended)
- **Docker** (optional - includes PostgreSQL, or install PostgreSQL separately)

## Quick Start

### 1. Initial Setup

```bash
# Copy environment variables
cp .env.example .env

# Create and activate Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```
Edit `.env` and update passwords and settings as needed.

### 2. Start Database (New Terminal)

**Option A: Using Docker (Recommended)**
```bash
docker-compose up
```
*Add `-d` flag to run in background: `docker-compose up -d`*

**Option B: Local PostgreSQL**
- Install PostgreSQL and ensure it's running on port 5433

*The backend will automatically create the database on first run*

### 3. Start Backend (New Terminal)

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
uvicorn backend.main:app --reload
```

### 4. Start Frontend (New Terminal)

```bash
cd frontend
npm install
npm run dev
```

### 5. Access Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 6. Database Migration

The dwell-time and zone-counting modules require the new tables defined in `backend/migrations/002_dwell_and_counter_tables.sql`.

If you are using Docker, the migration runs automatically on first boot. For manual setups:

```bash
psql "$DATABASE_URL" -f backend/migrations/002_dwell_and_counter_tables.sql
```

> Replace `$DATABASE_URL` with the same connection string configured in `.env`.

## New Modules & APIs

### Dwell Time Tracking

- **Frontend**: Navigate to `/dwell-time`
- **Workflow**:
  1. Select a camera, draw custom zones on the live feed.
  2. Upload a face image to enroll a target and pick the zones to monitor.
  3. Watch live timers, arrival/exit timestamps, and historical dwell logs.
- **Key Endpoints**:
  - `POST /dwell/targets` (multipart) – enroll a face and store embeddings
  - `GET /dwell/live` – list active sessions with running timers
  - `GET /dwell/sessions` – fetch dwell history (filterable by target/zone)

### Multi-Object Zone Counting

- **Frontend**: Navigate to `/zone-counting`
- **Workflow**:
  1. Draw counting zones per camera (works for any YOLO class, not just people).
  2. View per-zone cards showing current presence + entry/exit totals per class.
  3. Review chronological event logs and export them as CSV/JSON.
- **Key Endpoints**:
  - `GET /zone-counters/live` – aggregated stats + live occupancies per zone
  - `GET /zone-counters/events` – paged, filterable event history
  - `GET /zone-counters/export?format=csv|json` – download analytics

Both modules reuse the same smooth zone-drawing tool (React + SVG overlay) and store data in PostgreSQL for durability.

## Project Structure

```
.
├── backend/          # FastAPI backend
├── frontend/         # React + Vite frontend  
├── prometheus/       # Monitoring configuration
├── docker-compose.yml
├── requirements.txt  # Python dependencies
├── .env.example     # Environment variables template
└── README.md
```

## Troubleshooting

- **Database Connection**: Check your `.env` file credentials match your PostgreSQL setup
- **Port Conflicts**: Update ports in `.env` or `docker-compose.yml` if needed
- **Dependencies**: Ensure Python 3.10+ and Node.js 14+ are installed

## License

MIT License - see [LICENSE](LICENSE) file for details.
