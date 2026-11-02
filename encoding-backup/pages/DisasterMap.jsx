import { useMemo, useState } from "react";
import { MapContainer, TileLayer, Rectangle, Popup, useMap } from "react-leaflet";
import { MapPin, Layers, Info, Crosshair } from "lucide-react";
import "leaflet/dist/leaflet.css";

function GridView({ latitude, longitude }) {
  const map = useMap();
  const south = Math.floor(latitude / 2) * 2;
  const west = Math.floor(longitude / 2) * 2;

  useMemo(() => {
    map.setView([latitude, longitude], 5);
  }, [latitude, longitude, map]);

  return (
    <Rectangle
      bounds={[[south, west], [south + 2, west + 2]]}
      pathOptions={{ color: "#43c7ae", weight: 2, fillOpacity: 0.17 }}
    >
      <Popup>
        Selected 2° × 2° grid cell<br />
        Latitude: {south}° to {south + 2}°<br />
        Longitude: {west}° to {west + 2}°
      </Popup>
    </Rectangle>
  );
}

export default function DisasterMap() {
  const [latitude, setLatitude] = useState(20.5);
  const [longitude, setLongitude] = useState(79.2);

  const south = Math.floor(latitude / 2) * 2;
  const west = Math.floor(longitude / 2) * 2;

  return (
    <main className="intel-page">
      <div className="intel-heading">
        <div>
          <span className="intel-eyebrow">GEOSPATIAL INTELLIGENCE</span>
          <h1>Disaster Map</h1>
          <p>Explore geographical grid cells used by the earthquake analysis model.</p>
        </div>
        <span className="intel-tag"><MapPin size={15} /> 2° GRID RESOLUTION</span>
      </div>

      <section className="intel-panel map-controls">
        <div>
          <h3>Explore a geographical location</h3>
          <p>Select coordinates to view the corresponding model grid cell.</p>
        </div>
        <div className="map-coordinate-inputs">
          <label>Latitude
            <input type="number" min="-90" max="90" step="any" value={latitude}
              onChange={e => setLatitude(e.target.value === "" ? "" : Number(e.target.value))} />
          </label>
          <label>Longitude
            <input type="number" min="-180" max="180" step="any" value={longitude}
              onChange={e => setLongitude(e.target.value === "" ? "" : Number(e.target.value))} />
          </label>
          <button onClick={() => { setLatitude(20.5); setLongitude(79.2); }}>
            <Crosshair size={16} /> Reset
          </button>
        </div>
      </section>

      <section className="map-layout">
        <div className="intel-panel map-container">
          <div className="map-panel-heading">
            <h3><Layers size={18} /> Geographic grid explorer</h3>
            <span>OpenStreetMap</span>
          </div>
          {Number.isFinite(latitude) && latitude >= -90 && latitude <= 90 &&
           Number.isFinite(longitude) && longitude >= -180 && longitude <= 180 ? (
            <MapContainer center={[latitude, longitude]} zoom={5} scrollWheelZoom className="disaster-leaflet-map">
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              <GridView latitude={latitude} longitude={longitude} />
            </MapContainer>
          ) : (
            <div className="map-empty">Enter valid latitude and longitude values to explore the grid.</div>
          )}
        </div>

        <aside className="intel-panel map-details">
          <div className="intel-icon"><MapPin size={21} /></div>
          <h3>Selected grid cell</h3>
          <p className="intel-muted">Model spatial resolution: 2° × 2°</p>
          <div className="map-detail-item"><span>Latitude range</span><strong>{south}° to {south + 2}°</strong></div>
          <div className="map-detail-item"><span>Longitude range</span><strong>{west}° to {west + 2}°</strong></div>
          <div className="map-detail-item"><span>Grid centre</span><strong>{(south + 1).toFixed(1)}°, {(west + 1).toFixed(1)}°</strong></div>
          <div className="map-information">
            <Info size={17} />
            <span>The highlighted rectangle represents the selected model grid cell. This map does not currently display live earthquake events or a real-time risk heatmap.</span>
          </div>
        </aside>
      </section>

      <section className="intel-panel map-method-note">
        <h3>How the geographical model works</h3>
        <p>The earthquake classifier maps coordinates into 2° × 2° grid cells. Its target is whether at least one M4+ earthquake is recorded in the next calendar month. This is a retrospective research classifier, not an operational earthquake early-warning system.</p>
      </section>
    </main>
  );
}
