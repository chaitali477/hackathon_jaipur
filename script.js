// Global variables
let riskMap;
let simulationMap;
let deploymentMap;
let isSimulationRunning = false;
let simulationTime = 0;
let simulationInterval;
let fireSpreadLayers = [];
let deploymentLayers = [];

// ML API endpoints
const ML_API_BASE = window.location.origin.replace(':5000', ':5001');
let mlPredictions = {};
let realTimeUpdates = false;
let currentOptimization = null;

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeMaps();
    initializeCharts();
    initializeSimulation();

    // Initialize ML components
    initializeMLIntegration();

    // Initialize resource optimization
    initializeResourceOptimization();

    // Initialize environmental impact features
    initializeEnvironmentalImpact();

    // Initialize monitoring stats
    updateMonitoringStats();

    // Initialize fire theme effects
    initializeForestFireTheme();
    initializeFireInteractions();

    startDataUpdates();
});

// Fire theme initialization
function initializeForestFireTheme() {
    // Add fire effect overlay with animated particles
    const fireOverlay = document.createElement('div');
    fireOverlay.id = 'fire-overlay';
    fireOverlay.style.position = 'fixed';
    fireOverlay.style.top = '0';
    fireOverlay.style.left = '0';
    fireOverlay.style.width = '100%';
    fireOverlay.style.height = '100%';
    fireOverlay.style.pointerEvents = 'none';
    fireOverlay.style.zIndex = '1';
    fireOverlay.style.background = `
        radial-gradient(ellipse 400px 200px at 10% 90%, rgba(255, 69, 0, 0.1) 0%, transparent 60%),
        radial-gradient(ellipse 300px 150px at 90% 85%, rgba(255, 140, 0, 0.08) 0%, transparent 70%),
        radial-gradient(ellipse 200px 100px at 60% 80%, rgba(255, 69, 0, 0.06) 0%, transparent 80%)
    `;
    fireOverlay.style.opacity = '0.4';
    fireOverlay.style.animation = 'fireGlow 6s ease-in-out infinite alternate';

    document.body.appendChild(fireOverlay);

    // Create floating embers
    createFloatingEmbers();

    // Simulate dynamic fire intensity
    setInterval(() => {
        const intensity = Math.random() * 0.2 + 0.3; // Vary between 30% and 50% opacity
        fireOverlay.style.opacity = intensity;
    }, 4000);
}

function createFloatingEmbers() {
    setInterval(() => {
        if (Math.random() < 0.3) { // 30% chance to create ember
            const ember = document.createElement('div');
            ember.style.position = 'fixed';
            ember.style.width = '3px';
            ember.style.height = '3px';
            ember.style.background = Math.random() > 0.5 ? '#FF4500' : '#FF8C00';
            ember.style.borderRadius = '50%';
            ember.style.left = Math.random() * window.innerWidth + 'px';
            ember.style.top = window.innerHeight + 'px';
            ember.style.pointerEvents = 'none';
            ember.style.zIndex = '2';
            ember.style.boxShadow = `0 0 6px ${ember.style.background}`;
            ember.style.opacity = '0.8';
            ember.style.animation = 'emberFloat 8s linear forwards';

            document.body.appendChild(ember);

            setTimeout(() => {
                if (document.body.contains(ember)) {
                    document.body.removeChild(ember);
                }
            }, 8000);
        }
    }, 2000);
}

// Fire cursor trail effect
let fireParticles = [];

function initializeFireInteractions() {
    // Initialize fire cursor trail
    document.addEventListener('mousemove', function(e) {
        if (Math.random() < 0.1) { // Only create particles occasionally for performance
            createFireParticle(e.clientX, e.clientY);
        }
    });

    // Initialize fire click effects
    initializeFireClickRipples();
}

function createFireParticle(x, y) {
    const particle = document.createElement('div');
    particle.className = 'fire-cursor-trail';
    particle.style.left = x + 'px';
    particle.style.top = y + 'px';
    particle.style.width = '8px';
    particle.style.height = '8px';
    particle.style.background = `radial-gradient(circle, ${Math.random() > 0.5 ? '#FF4500' : '#FF8C00'}, transparent)`;

    document.body.appendChild(particle);

    setTimeout(() => {
        if (document.body.contains(particle)) {
            document.body.removeChild(particle);
        }
    }, 1000);
}

// Fire click ripples
function initializeFireClickRipples() {
    document.addEventListener('click', function(e) {
        const ripple = document.createElement('div');
        ripple.style.position = 'fixed';
        ripple.style.left = e.clientX + 'px';
        ripple.style.top = e.clientY + 'px';
        ripple.style.width = '0px';
        ripple.style.height = '0px';
        ripple.style.background = 'radial-gradient(circle, rgba(255, 69, 0, 0.6), transparent)';
        ripple.style.borderRadius = '50%';
        ripple.style.pointerEvents = 'none';
        ripple.style.zIndex = '9998';
        ripple.style.transform = 'translate(-50%, -50%)';
        ripple.style.animation = 'fireClickRipple 0.8s ease-out forwards';

        document.body.appendChild(ripple);

        setTimeout(() => {
            if (document.body.contains(ripple)) {
                document.body.removeChild(ripple);
            }
        }, 800);
    });
}

// Navigation functionality
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');

    // Smooth scroll navigation
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);

            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth' });

                // Update active nav link
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            }
        });
    });

    // Update active nav on scroll
    window.addEventListener('scroll', () => {
        const sections = document.querySelectorAll('.section');
        const scrollPos = window.scrollY + 100;

        sections.forEach(section => {
            const top = section.offsetTop;
            const bottom = top + section.offsetHeight;
            const id = section.getAttribute('id');

            if (scrollPos >= top && scrollPos <= bottom) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    });
}

// Initialize maps
function initializeMaps() {
    // Check if Leaflet is loaded
    if (typeof L === 'undefined') {
        console.error('Leaflet library not loaded');
        return;
    }

    try {
        // Fire Risk Map
        const riskMapElement = document.getElementById('risk-map');
        if (riskMapElement) {
            riskMap = L.map('risk-map').setView([30.0668, 79.0193], 8);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(riskMap);

            // Add risk zones for Uttarakhand districts
            addRiskZones();
        }

        // Simulation Map
        const simulationMapElement = document.getElementById('simulation-map');
        if (simulationMapElement) {
            simulationMap = L.map('simulation-map').setView([30.0668, 79.0193], 8);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(simulationMap);

            // Add click listener for fire simulation
            simulationMap.on('click', function(e) {
                startFireSimulation(e.latlng);
            });

            // Add forest areas
            addForestAreas();
        }

        // Deployment Map for Resource Optimization
        const deploymentMapElement = document.getElementById('deployment-map');
        if (deploymentMapElement) {
            deploymentMap = L.map('deployment-map').setView([30.0668, 79.0193], 7);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(deploymentMap);

            // Add district boundaries
            addDistrictBoundaries();
        }

        // Initialize search functionality for maps
        initializeMapSearch();

        console.log('Maps initialized successfully');
    } catch (error) {
        console.error('Error initializing maps:', error);
    }
}

// Add risk zones to the map
function addRiskZones() {
    if (!riskMap) return;

    const riskZones = [
        {
            name: 'Nainital District',
            coords: [[29.2, 79.3], [29.6, 79.3], [29.6, 79.8], [29.2, 79.8]],
            risk: 'very-high',
            color: '#ff4444'
        },
        {
            name: 'Almora District',
            coords: [[29.5, 79.5], [29.9, 79.5], [29.9, 80.0], [29.5, 80.0]],
            risk: 'high',
            color: '#ffa726'
        },
        {
            name: 'Dehradun District',
            coords: [[30.1, 77.8], [30.5, 77.8], [30.5, 78.3], [30.1, 78.3]],
            risk: 'moderate',
            color: '#66bb6a'
        }
    ];

    riskZones.forEach(zone => {
        const polygon = L.polygon(zone.coords, {
            color: zone.color,
            fillColor: zone.color,
            fillOpacity: 0.4
        }).addTo(riskMap);

        polygon.bindPopup(`
            <div>
                <h4>${zone.name}</h4>
                <p>Risk Level: ${zone.risk.replace('-', ' ').toUpperCase()}</p>
            </div>
        `);
    });
}

// Add forest areas to simulation map
function addForestAreas() {
    if (!simulationMap) return;

    const forestAreas = [
        {
            name: 'Jim Corbett National Park',
            coords: [[29.4, 78.7], [29.7, 78.7], [29.7, 79.1], [29.4, 79.1]],
            color: '#2d5a2d'
        },
        {
            name: 'Valley of Flowers',
            coords: [[30.7, 79.5], [30.8, 79.5], [30.8, 79.7], [30.7, 79.7]],
            color: '#2d5a2d'
        }
    ];

    forestAreas.forEach(forest => {
        const polygon = L.polygon(forest.coords, {
            color: forest.color,
            fillColor: forest.color,
            fillOpacity: 0.6
        }).addTo(simulationMap);

        polygon.bindPopup(`<h4>${forest.name}</h4>`);
    });
}

// Initialize charts
function initializeCharts() {
    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.error('Chart.js library not loaded');
        return;
    }

    try {
        // Performance Chart (Original)
        const performanceCtx = document.getElementById('performanceChart');
        if (performanceCtx) {
            new Chart(performanceCtx.getContext('2d'), {
                type: 'doughnut',
                data: {
                    labels: ['Accurate Predictions', 'False Positives'],
                    datasets: [{
                        data: [97, 3],
                        backgroundColor: ['#66bb6a', '#ff6b35'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#ffffff'
                            }
                        }
                    }
                }
            });
        }

        // Initialize other charts
        initializeTimelineChart();
        initializeFireSpreadChart();
        initializeGaugeCharts();
        initializeAlertStatsChart();

        console.log('Charts initialized successfully');
    } catch (error) {
        console.error('Error initializing charts:', error);
    }
}

function initializeTimelineChart() {
    const riskTimelineCtx = document.getElementById('riskTimelineChart');
    if (!riskTimelineCtx) return;

    const riskTimelineChart = new Chart(riskTimelineCtx.getContext('2d'), {
        type: 'line',
        data: {
            labels: ['6AM', '9AM', '12PM', '3PM', '6PM', '9PM', '12AM', '3AM'],
            datasets: [
                {
                    label: 'Dehradun',
                    data: [25, 35, 55, 75, 85, 65, 45, 30],
                    borderColor: '#66bb6a',
                    backgroundColor: 'rgba(102, 187, 106, 0.1)',
                    fill: false,
                    tension: 0.4
                },
                {
                    label: 'Nainital',
                    data: [45, 55, 70, 85, 90, 80, 60, 50],
                    borderColor: '#ff4444',
                    backgroundColor: 'rgba(255, 68, 68, 0.1)',
                    fill: false,
                    tension: 0.4
                },
                {
                    label: 'Haridwar',
                    data: [30, 40, 60, 70, 75, 55, 40, 35],
                    borderColor: '#ffa726',
                    backgroundColor: 'rgba(255, 167, 38, 0.1)',
                    fill: false,
                    tension: 0.4
                },
                {
                    label: 'Rishikesh',
                    data: [20, 30, 45, 65, 70, 50, 35, 25],
                    borderColor: '#42a5f5',
                    backgroundColor: 'rgba(66, 165, 245, 0.1)',
                    fill: false,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#ffffff'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#ffffff'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    min: 0,
                    max: 100,
                    ticks: {
                        color: '#ffffff',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        }
    });

    // Store chart reference
    if (!window.chartInstances) {
        window.chartInstances = {};
    }
    window.chartInstances.riskTimeline = riskTimelineChart;
}

function initializeFireSpreadChart() {
    const fireSpreadCtx = document.getElementById('fireSpreadChart');
    if (!fireSpreadCtx) return;

    const fireSpreadChart = new Chart(fireSpreadCtx.getContext('2d'), {
        type: 'line',
        data: {
            labels: ['0h', '1h', '2h', '3h', '4h', '5h', '6h'],
            datasets: [{
                label: 'Area Burned (hectares)',
                data: [0, 12, 35, 78, 145, 225, 320],
                borderColor: '#ff6b35',
                backgroundColor: 'rgba(255, 107, 53, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#ffffff'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return 'Burned: ' + context.parsed.y + ' hectares';
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#ffffff'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    ticks: {
                        color: '#ffffff',
                        callback: function(value) {
                            return value + ' ha';
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        }
    });

    if (!window.chartInstances) {
        window.chartInstances = {};
    }
    window.chartInstances.fireSpread = fireSpreadChart;
}

function initializeGaugeCharts() {
    // Accuracy Gauge
    const accuracyCtx = document.getElementById('accuracyGauge');
    if (accuracyCtx) {
        new Chart(accuracyCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [97, 3],
                    backgroundColor: ['#66bb6a', 'rgba(255, 255, 255, 0.1)'],
                    borderWidth: 0,
                    cutout: '75%'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                }
            }
        });
    }

    // Uptime Gauge
    const uptimeCtx = document.getElementById('uptimeGauge');
    if (uptimeCtx) {
        new Chart(uptimeCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [99.8, 0.2],
                    backgroundColor: ['#66bb6a', 'rgba(255, 255, 255, 0.1)'],
                    borderWidth: 0,
                    cutout: '75%'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                }
            }
        });
    }

    // Speed Gauge
    const speedCtx = document.getElementById('speedGauge');
    if (speedCtx) {
        new Chart(speedCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [85, 15],
                    backgroundColor: ['#ffa726', 'rgba(255, 255, 255, 0.1)'],
                    borderWidth: 0,
                    cutout: '75%'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                }
            }
        });
    }
}

function initializeAlertStatsChart() {
    const alertStatsCtx = document.getElementById('alertStatsChart');
    if (!alertStatsCtx) return;

    new Chart(alertStatsCtx.getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: ['Fire Risk Warnings', 'Active Fire Detected', 'Evacuation Alerts', 'All Clear/Safe Zones'],
            datasets: [{
                data: [35, 25, 20, 20],
                backgroundColor: ['#ffa726', '#ff4444', '#ff6b35', '#66bb6a'],
                borderWidth: 2,
                borderColor: 'rgba(255, 255, 255, 0.1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#ffffff',
                        padding: 15,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return context.label + ': ' + percentage + '%';
                        }
                    }
                }
            }
        }
    });
}

// Initialize simulation controls
function initializeSimulation() {
    const playBtn = document.getElementById('play-simulation');
    const pauseBtn = document.getElementById('pause-simulation');
    const resetBtn = document.getElementById('reset-simulation');
    const speedSlider = document.getElementById('speed-slider');
    const speedValue = document.getElementById('speed-value');

    if (playBtn) {
        playBtn.addEventListener('click', () => {
            if (!isSimulationRunning) {
                startSimulation();
            }
        });
    }

    if (pauseBtn) {
        pauseBtn.addEventListener('click', pauseSimulation);
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', resetSimulation);
    }

    if (speedSlider && speedValue) {
        speedSlider.addEventListener('input', (e) => {
            const speed = e.target.value;
            speedValue.textContent = `${speed}x`;
            updateSimulationSpeed(speed);
        });
    }

    // Toggle prediction button
    const toggleBtn = document.getElementById('toggle-prediction');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', togglePrediction);
    }

    // Initialize simulation monitoring chart
    initializeSimulationMonitoringChart();
}

// Fire simulation functions
async function startFireSimulation(latlng) {
    if (!simulationMap) return;

    // Try ML-powered simulation first
    const mlSimulation = await simulateFireWithML(latlng);

    // Add animated fire origin marker with enhanced visuals
    const fireMarker = L.marker([latlng.lat, latlng.lng], {
        icon: L.divIcon({
            className: 'fire-marker origin-fire',
            html: `
                <div class="fire-icon-container">
                    <i class="fas fa-fire fire-flame"></i>
                    <div class="fire-glow"></div>
                    <div class="fire-sparks"></div>
                </div>
            `,
            iconSize: [40, 40],
            iconAnchor: [20, 20]
        })
    }).addTo(simulationMap);

    fireSpreadLayers.push(fireMarker);

    // Store ML simulation data if available
    if (mlSimulation) {
        window.currentMLSimulation = mlSimulation;
        showToast('AI-powered simulation active', 'success');
    }

    // Add initial burn circle
    const initialBurn = L.circle([latlng.lat, latlng.lng], {
        color: '#ff6b35',
        fillColor: '#ff4444',
        fillOpacity: 0.7,
        radius: 200,
        className: 'fire-burn-area'
    }).addTo(simulationMap);

    fireSpreadLayers.push(initialBurn);
}

function startSimulation() {
    isSimulationRunning = true;
    const speedSlider = document.getElementById('speed-slider');
    const speed = speedSlider ? parseInt(speedSlider.value) : 1;

    // Limit minimum interval to prevent overwhelming the browser
    const minInterval = 500; // Minimum 500ms between updates
    const interval = Math.max(minInterval, 2000 / speed);

    simulationInterval = setInterval(() => {
        simulationTime += 1;
        updateSimulationTime();

        // Update monitoring stats
        updateMonitoringStats();

        // Use requestAnimationFrame for smoother performance
        requestAnimationFrame(() => {
            simulateFireSpread();
        });
    }, interval);

    // Initial update
    updateMonitoringStats();
}

function pauseSimulation() {
    isSimulationRunning = false;
    if (simulationInterval) {
        clearInterval(simulationInterval);
    }
}

function resetSimulation() {
    pauseSimulation();
    simulationTime = 0;
    updateSimulationTime();

    // Reset monitoring stats
    updateMonitoringStats();

    // Clear fire spread layers
    if (simulationMap) {
        fireSpreadLayers.forEach(layer => {
            try {
                simulationMap.removeLayer(layer);
            } catch (e) {
                // Ignore errors for layers already removed
            }
        });
    }
    fireSpreadLayers = [];
}

function updateSimulationSpeed(speed) {
    if (isSimulationRunning) {
        pauseSimulation();
        startSimulation();
    }
}

function simulateFireSpread() {
    // Optimized fire spread simulation
    if (fireSpreadLayers.length > 0 && simulationMap) {
        // Limit processing to prevent overwhelming the browser
        if (fireSpreadLayers.length > 100) {
            const excessLayers = fireSpreadLayers.splice(0, 20);
            excessLayers.forEach(layer => {
                try {
                    simulationMap.removeLayer(layer);
                } catch (e) {
                    // Ignore errors for layers already removed
                }
            });
        }

        // Get environmental parameters
        const windSpeed = getElementValue('wind-speed', 15);
        const windDirection = getElementText('wind-direction', 'NE');
        const temperature = getElementValue('temperature', 32);
        const humidity = getElementValue('humidity', 45);

        // Convert wind direction to angle
        const windAngles = {
            'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
            'S': 180, 'SW': 225, 'W': 270, 'NW': 315
        };
        const windAngle = (windAngles[windDirection] || 0) * Math.PI / 180;

        // Limit the number of fire sources processed per cycle
        const fireSources = fireSpreadLayers.filter(layer => 
            layer instanceof L.Marker && 
            layer.options.icon && 
            layer.options.icon.options.className && 
            layer.options.icon.options.className.includes('fire-marker')
        ).slice(-10);

        let newSpreadCount = 0;
        const maxNewSpreads = 3;

        fireSources.forEach((fireSource) => {
            if (newSpreadCount >= maxNewSpreads) return;

            if (Math.random() < 0.4) {
                const sourceLatlng = fireSource.getLatLng();

                // Simplified spread calculation
                const baseSpread = 0.005;
                const windFactor = windSpeed / 20;
                const tempFactor = temperature / 35;
                const humidityFactor = (100 - humidity) / 120;

                const spreadDistance = baseSpread * windFactor * tempFactor * humidityFactor;

                // Calculate new position
                const spreadAngle = windAngle + (Math.random() - 0.5) * Math.PI / 3;
                const newLat = sourceLatlng.lat + Math.cos(spreadAngle) * spreadDistance;
                const newLng = sourceLatlng.lng + Math.sin(spreadAngle) * spreadDistance;

                // Create simplified fire marker
                const spreadMarker = L.marker([newLat, newLng], {
                    icon: L.divIcon({
                        className: 'fire-marker spread-fire',
                        html: '<i class="fas fa-fire fire-flame"></i>',
                        iconSize: [20, 20],
                        iconAnchor: [10, 10]
                    })
                }).addTo(simulationMap);

                // Create smaller burn area
                const burnRadius = 100 + Math.random() * 50;
                const burnArea = L.circle([newLat, newLng], {
                    color: '#ff4444',
                    fillColor: '#cc0000',
                    fillOpacity: 0.3,
                    radius: burnRadius,
                    weight: 1
                }).addTo(simulationMap);

                fireSpreadLayers.push(spreadMarker);
                fireSpreadLayers.push(burnArea);
                newSpreadCount++;
            }
        });
    }
}

function updateSimulationTime() {
    const timeElement = document.getElementById('simulation-time');
    if (timeElement) {
        const hours = Math.floor(simulationTime / 60);
        const minutes = simulationTime % 60;
        timeElement.textContent = 
            hours > 0 ? `${hours}h ${minutes}m` : `${minutes} minutes`;
    }
}

// Helper functions
function getElementValue(id, defaultValue) {
    const element = document.getElementById(id);
    if (element) {
        const value = parseInt(element.textContent);
        return isNaN(value) ? defaultValue : value;
    }
    return defaultValue;
}

function getElementText(id, defaultValue) {
    const element = document.getElementById(id);
    return element ? element.textContent : defaultValue;
}

// Toggle prediction functionality
function togglePrediction() {
    const btn = document.getElementById('toggle-prediction');
    if (!btn) return;

    const isNextDay = btn.textContent.includes('Current');

    if (isNextDay) {
        btn.innerHTML = '<i class="fas fa-clock"></i> Show Next Day Prediction';
        showCurrentDayPrediction();
    } else {
        btn.innerHTML = '<i class="fas fa-calendar"></i> Show Current Day';
        showNextDayPrediction();
    }
}

function showCurrentDayPrediction() {
    updateRiskZones([
        { name: 'Nainital District', risk: 'very-high', percentage: 85 },
        { name: 'Almora District', risk: 'high', percentage: 68 },
        { name: 'Dehradun District', risk: 'moderate', percentage: 42 }
    ]);

    updateMapRiskColors('current');

    const lastUpdateElement = document.getElementById('last-update');
    if (lastUpdateElement) {
        lastUpdateElement.textContent = '2 minutes ago';
    }
}

function showNextDayPrediction() {
    updateRiskZones([
        { name: 'Nainital District', risk: 'very-high', percentage: 92 },
        { name: 'Almora District', risk: 'very-high', percentage: 78 },
        { name: 'Dehradun District', risk: 'high', percentage: 65 }
    ]);

    updateMapRiskColors('predicted');

    const lastUpdateElement = document.getElementById('last-update');
    if (lastUpdateElement) {
        lastUpdateElement.textContent = 'Predicted for tomorrow';
    }
}

function updateRiskZones(zones) {
    const riskContainer = document.querySelector('.risk-zones');
    if (!riskContainer) return;

    riskContainer.innerHTML = '';

    zones.forEach(zone => {
        const riskClass = zone.risk === 'very-high' ? 'high-risk' : 
                         zone.risk === 'high' ? 'moderate-risk' : 'low-risk';

        const riskItem = document.createElement('div');
        riskItem.className = `risk-item ${riskClass}`;
        riskItem.innerHTML = `
            <div class="risk-color"></div>
            <div class="risk-info">
                <span class="risk-level">${zone.risk.replace('-', ' ').toUpperCase()} Risk</span>
                <span class="risk-area">${zone.name}</span>
            </div>
            <div class="risk-percentage">${zone.percentage}%</div>
        `;
        riskContainer.appendChild(riskItem);
    });
}

function updateMapRiskColors(type) {
    if (!riskMap) return;

    // Clear existing polygons
    riskMap.eachLayer(layer => {
        if (layer instanceof L.Polygon) {
            riskMap.removeLayer(layer);
        }
    });

    // Define zones based on prediction type
    const zones = type === 'current' ? [
        {
            name: 'Nainital District',
            coords: [[29.2, 79.3], [29.6, 79.3], [29.6, 79.8], [29.2, 79.8]],
            risk: 'very-high',
            color: '#ff4444'
        },
        {
            name: 'Almora District',
            coords: [[29.5, 79.5], [29.9, 79.5], [29.9, 80.0], [29.5, 80.0]],
            risk: 'high',
            color: '#ffa726'
        },
        {
            name: 'Dehradun District',
            coords: [[30.1, 77.8], [30.5, 77.8], [30.5, 78.3], [30.1, 78.3]],
            risk: 'moderate',
            color: '#66bb6a'
        }
    ] : [
        {
            name: 'Nainital District',
            coords: [[29.2, 79.3], [29.6, 79.3], [29.6, 79.8], [29.2, 79.8]],
            risk: 'very-high',
            color: '#cc0000'
        },
        {
            name: 'Almora District',
            coords: [[29.5, 79.5], [29.9, 79.5], [29.9, 80.0], [29.5, 80.0]],
            risk: 'very-high',
            color: '#ff4444'
        },
        {
            name: 'Dehradun District',
            coords: [[30.1, 77.8], [30.5, 77.8], [30.5, 78.3], [30.1, 78.3]],
            risk: 'high',
            color: '#ffa726'
        }
    ];

    // Add updated zones to map
    zones.forEach(zone => {
        const polygon = L.polygon(zone.coords, {
            color: zone.color,
            fillColor: zone.color,
            fillOpacity: type === 'predicted' ? 0.6 : 0.4,
            weight: type === 'predicted' ? 3 : 2
        }).addTo(riskMap);

        const riskLevel = zone.risk.replace('-', ' ').toUpperCase();
        const prefix = type === 'predicted' ? 'Predicted: ' : '';

        polygon.bindPopup(`
            <div>
                <h4>${zone.name}</h4>
                <p>${prefix}Risk Level: ${riskLevel}</p>
            </div>
        `);
    });
}

// Enhanced Search Functionality
function initializeMapSearch() {
    const searchInput = document.querySelector('.search-input');

    const searchMap = (query, map) => {
        const locations = {
            'nainital': { lat: 29.3806, lng: 79.4422 },
            'almora': { lat: 29.6500, lng: 79.6667 },
            'dehradun': { lat: 30.3165, lng: 78.0322 },
            'haridwar': { lat: 29.9457, lng: 78.1642 },
            'rishikesh': { lat: 30.0869, lng: 78.2676 },
            'uttarakhand': { lat: 30.0668, lng: 79.0193 },
            'jim corbett': { lat: 29.5308, lng: 78.9514 },
            'corbett': { lat: 29.5308, lng: 78.9514 },
            'valley of flowers': { lat: 30.7268, lng: 79.6045 },
            'chamoli': { lat: 30.4000, lng: 79.3200 },
            'pithoragarh': { lat: 29.5833, lng: 80.2167 },
            'tehri': { lat: 30.3900, lng: 78.4800 },
            'pauri': { lat: 30.1500, lng: 78.7800 },
            'rudraprayag': { lat: 30.2800, lng: 78.9800 },
            'bageshwar': { lat: 29.8400, lng: 79.7700 },
            'champawat': { lat: 29.3400, lng: 80.0900 },
            'uttarkashi': { lat: 30.7300, lng: 78.4500 },
            'udham singh nagar': { lat: 28.9750, lng: 79.4000 }
        };

        const lowerCaseQuery = query.toLowerCase();
        if (locations[lowerCaseQuery]) {
            const { lat, lng } = locations[lowerCaseQuery];
            map.setView([lat, lng], 12);
            L.marker([lat, lng]).addTo(map).bindPopup(query.charAt(0).toUpperCase() + query.slice(1)).openPopup();
            showToast(`Navigated to ${query}`, 'success');
            return true;
        } else {
            showToast(`Location "${query}" not found.`, 'error');
            return false;
        }
    };

    const performSearch = (query) => {
        if (query && riskMap && simulationMap) {
            const riskSuccess = searchMap(query, riskMap);
            const simulationSuccess = searchMap(query, simulationMap);

            if (riskSuccess || simulationSuccess) {
                setTimeout(() => {
                    const fireRiskSection = document.getElementById('fire-risk');
                    if (fireRiskSection) {
                        fireRiskSection.scrollIntoView({ 
                            behavior: 'smooth',
                            block: 'start'
                        });

                        const mapContainers = document.querySelectorAll('.map-container');
                        mapContainers.forEach(container => {
                            container.style.boxShadow = '0 0 30px rgba(255, 107, 53, 0.8)';
                            container.style.transform = 'scale(1.01)';
                            container.style.transition = 'all 0.5s ease';
                        });

                        setTimeout(() => {
                            mapContainers.forEach(container => {
                                container.style.boxShadow = '';
                                container.style.transform = '';
                            });
                        }, 2000);
                    }
                }, 500);
            }

            if (searchInput) {
                searchInput.value = '';
            }
        }
    };

    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                const query = searchInput.value.trim();
                performSearch(query);
            }
        });
    }
}

// Start real-time data updates
function startDataUpdates() {
    setInterval(updateEnvironmentalData, 30000);
    setInterval(updateAlerts, 60000);
    setInterval(updateTimeStamps, 60000);
    setInterval(updateChartData, 45000);
    setInterval(updateFireSpreadChart, 10000);
    setInterval(updateActivityFeed, 45000);
    setInterval(updateEnvironmentalConditions, 35000);
}

function updateEnvironmentalData() {
    const windSpeed = Math.floor(Math.random() * 20) + 5;
    const temperature = Math.floor(Math.random() * 15) + 25;
    const humidity = Math.floor(Math.random() * 40) + 30;

    updateElementText('wind-speed', `${windSpeed} km/h`);
    updateElementText('temperature', `${temperature}°C`);
    updateElementText('humidity', `${humidity}%`);

    const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    const randomDirection = directions[Math.floor(Math.random() * directions.length)];
    updateElementText('wind-direction', randomDirection);
}

function updateElementText(id, text) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = text;
    }
}

function updateAlerts() {
    const alertTimes = document.querySelectorAll('.alert-time');
    alertTimes.forEach((timeEl, index) => {
        const baseTime = (index + 1) * 15;
        timeEl.textContent = `${baseTime} minutes ago`;
    });
}

function updateTimeStamps() {
    const lastUpdateElement = document.getElementById('last-update');
    if (lastUpdateElement) {
        lastUpdateElement.textContent = 'Just now';
        setTimeout(() => {
            lastUpdateElement.textContent = '1 minute ago';
        }, 5000);
    }
}

function updateChartData() {
    if (window.chartInstances && window.chartInstances.riskTimeline) {
        const chart = window.chartInstances.riskTimeline;
        chart.data.datasets.forEach((dataset) => {
            dataset.data = dataset.data.map(value => {
                const variation = (Math.random() - 0.5) * 10;
                return Math.max(0, Math.min(100, value + variation));
            });
        });
        chart.update('none');
    }

    updateGaugeValues();
    updateAlertStatistics();
}

function updateFireSpreadChart() {
    if (isSimulationRunning && window.chartInstances && window.chartInstances.fireSpread) {
        const chart = window.chartInstances.fireSpread;
        const lastValue = chart.data.datasets[0].data[chart.data.datasets[0].data.length - 1];

        const timeLabel = simulationTime + 'h';
        const newArea = lastValue + Math.random() * 50 + 20;

        if (chart.data.labels.length > 10) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }

        chart.data.labels.push(timeLabel);
        chart.data.datasets[0].data.push(Math.round(newArea));

        chart.update('none');
    }
}

function updateGaugeValues() {
    const newAccuracy = Math.max(95, Math.min(99, 97 + (Math.random() - 0.5) * 2));
    updateElementText('accuracyValue', newAccuracy.toFixed(1) + '%');

    const newUptime = Math.max(99.5, Math.min(100, 99.8 + (Math.random() - 0.5) * 0.3));
    updateElementText('uptimeValue', newUptime.toFixed(1) + '%');

    const newSpeed = Math.max(70, Math.min(95, 85 + (Math.random() - 0.5) * 10));
    updateElementText('speedValue', Math.round(newSpeed) + '%');
}

function updateAlertStatistics() {
    const totalAlerts = Math.floor(Math.random() * 20) + 130;
    const activeFires = Math.floor(Math.random() * 5) + 5;
    const responseTime = Math.floor(Math.random() * 8) + 8;

    updateElementText('totalAlerts', totalAlerts);
    updateElementText('activeFires', activeFires);
    updateElementText('responseTime', responseTime + ' min');
}

function updateActivityFeed() {
    const activities = [
        {
            icon: 'fas fa-satellite-dish',
            title: 'Satellite Data Updated',
            description: 'New MODIS imagery processed for Nainital region',
            time: '2 minutes ago'
        },
        {
            icon: 'fas fa-exclamation-triangle',
            title: 'Risk Level Updated',
            description: 'Almora District elevated to High Risk status',
            time: '8 minutes ago'
        },
        {
            icon: 'fas fa-cloud-sun',
            title: 'Weather Data Sync',
            description: 'ERA5 meteorological data synchronized',
            time: '15 minutes ago'
        }
    ];

    const activityFeed = document.querySelector('.activity-feed');
    if (activityFeed && Math.random() < 0.1) {
        const randomActivity = activities[Math.floor(Math.random() * activities.length)];
        const newActivityHtml = `
            <div class="activity-item new">
                <div class="activity-icon">
                    <i class="${randomActivity.icon}"></i>
                </div>
                <div class="activity-content">
                    <div class="activity-title">${randomActivity.title}</div>
                    <div class="activity-description">${randomActivity.description}</div>
                    <div class="activity-time">Just now</div>
                </div>
            </div>
        `;

        activityFeed.insertAdjacentHTML('afterbegin', newActivityHtml);

        const activityItems = activityFeed.querySelectorAll('.activity-item');
        if (activityItems.length > 5) {
            activityItems[activityItems.length - 1].remove();
        }

        activityItems.forEach((item, index) => {
            if (index > 0) {
                item.classList.remove('new');
            }
        });
    }
}

function updateEnvironmentalConditions() {
    const temperatureEl = document.querySelector('.condition-card.temperature .condition-value');
    const humidityEl = document.querySelector('.condition-card.humidity .condition-value');
    const windEl = document.querySelector('.condition-card.wind .condition-value');

    if (temperatureEl) {
        const newTemp = Math.floor(Math.random() * 8) + 28;
        temperatureEl.textContent = newTemp + '°C';
    }

    if (humidityEl) {
        const newHumidity = Math.floor(Math.random() * 30) + 35;
        humidityEl.textContent = newHumidity + '%';
    }

    if (windEl) {
        const newWind = Math.floor(Math.random() * 15) + 8;
        windEl.textContent = newWind + ' km/h';
    }

    if (Math.random() < 0.3) {
        updateMLPredictions();
    }
}

// ML Integration Functions
function initializeMLIntegration() {
    startMLRealTimeUpdates();
    loadMLModelInfo();
    setInterval(updateMLPredictions, 60000);
    showToast('AI/ML models initialized successfully', 'success');
}

async function startMLRealTimeUpdates() {
    try {
        const response = await fetch(`${ML_API_BASE}/api/ml/start-realtime`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            realTimeUpdates = true;
            showToast('Real-time AI predictions activated', 'success');
        }
    } catch (error) {
        console.warn('ML API not available, using fallback predictions');
        showToast('Using local AI predictions', 'warning');
    }
}

async function updateMLPredictions() {
    try {
        const envData = getCurrentEnvironmentalData();
        const response = await fetch(`${ML_API_BASE}/api/ml/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(envData)
        });

        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                mlPredictions = result.predictions;
                updateDashboardWithMLPredictions(result.predictions);
            }
        }
    } catch (error) {
        const envData = getCurrentEnvironmentalData();
        mlPredictions = generateFallbackPredictions(envData);
        updateDashboardWithMLPredictions(mlPredictions);
    }
}

function getCurrentEnvironmentalData() {
    const temperature = getElementValue('temperature', 32);
    const humidity = getElementValue('humidity', 45);
    const windSpeed = getElementValue('wind-speed', 15);
    const windDirection = getElementText('wind-direction', 'NE');

    return {
        temperature,
        humidity,
        wind_speed: windSpeed,
        wind_direction: windDirection,
        ndvi: 0.6 + (Math.random() - 0.5) * 0.2,
        elevation: 1500 + Math.random() * 500,
        slope: 10 + Math.random() * 20,
        vegetation_density: 'moderate'
    };
}

function updateDashboardWithMLPredictions(predictions) {
    if (predictions.ensemble_risk_score) {
        const accuracyEl = document.getElementById('accuracyValue');
        if (accuracyEl && predictions.confidence_interval) {
            const confidence = (predictions.confidence_interval.confidence_level * 100).toFixed(1);
            accuracyEl.textContent = confidence + '%';
        }
    }
}

function generateFallbackPredictions(envData) {
    const tempFactor = Math.min(envData.temperature / 40, 1);
    const humidityFactor = Math.max(0, (100 - envData.humidity) / 100);
    const windFactor = Math.min(envData.wind_speed / 30, 1);

    const baseRisk = (tempFactor * 0.4 + humidityFactor * 0.4 + windFactor * 0.2);
    const ensemble_risk = Math.min(baseRisk + Math.random() * 0.1, 1);

    return {
        ensemble_risk_score: ensemble_risk,
        ml_prediction: {
            overall_risk: ensemble_risk,
            confidence: 0.85,
            risk_category: ensemble_risk > 0.7 ? 'high' : ensemble_risk > 0.4 ? 'moderate' : 'low'
        },
        confidence_interval: {
            confidence_level: 0.85,
            lower_bound: Math.max(0, ensemble_risk - 0.1),
            upper_bound: Math.min(1, ensemble_risk + 0.1)
        }
    };
}

async function simulateFireWithML(latlng) {
    try {
        const envData = getCurrentEnvironmentalData();
        const response = await fetch(`${ML_API_BASE}/api/ml/simulate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                lat: latlng.lat,
                lng: latlng.lng,
                duration: 6,
                ...envData
            })
        });

        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                showToast('AI-powered fire simulation completed', 'success');
                return result.simulation;
            }
        }
    } catch (error) {
        console.warn('ML simulation failed, using fallback');
        showToast('Using simplified fire simulation', 'warning');
    }

    return null;
}

async function loadMLModelInfo() {
    try {
        const response = await fetch(`${ML_API_BASE}/api/ml/model-info`);
        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                const accuracyEl = document.getElementById('accuracyValue');
                if (accuracyEl && result.models.convlstm_unet.accuracy) {
                    accuracyEl.textContent = result.models.convlstm_unet.accuracy;
                }
                window.mlModelInfo = result.models;
            }
        }
    } catch (error) {
        console.warn('ML model info unavailable');
    }
}

// Initialize simulation monitoring chart
function initializeSimulationMonitoringChart() {
    const ctx = document.getElementById('simulationMonitoringChart');
    if (!ctx) return;

    const simulationMonitoringChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: ['0m'],
            datasets: [
                {
                    label: 'Area Burned (ha)',
                    data: [0],
                    borderColor: '#ff6b35',
                    backgroundColor: 'rgba(255, 107, 53, 0.1)',
                    fill: true,
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Fire Perimeter (km)',
                    data: [0],
                    borderColor: '#ffa726',
                    backgroundColor: 'rgba(255, 167, 38, 0.1)',
                    fill: false,
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#ffffff',
                        font: {
                            size: 11
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#ffffff',
                        font: {
                            size: 10
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    ticks: {
                        color: '#ffffff',
                        font: {
                            size: 10
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    title: {
                        display: true,
                        text: 'Area (ha)',
                        color: '#ff6b35',
                        font: {
                            size: 10
                        }
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    ticks: {
                        color: '#ffffff',
                        font: {
                            size: 10
                        }
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                    title: {
                        display: true,
                        text: 'Perimeter (km)',
                        color: '#ffa726',
                        font: {
                            size: 10
                        }
                    }
                }
            }
        }
    });

    if (!window.chartInstances) {
        window.chartInstances = {};
    }
    window.chartInstances.simulationMonitoring = simulationMonitoringChart;
}

// Update monitoring stats
function updateMonitoringStats() {
    if (isSimulationRunning) {
        const timeInHours = simulationTime / 60;
        const baseArea = Math.pow(timeInHours, 1.5) * 25;
        const burnedArea = baseArea + (Math.random() * 20 - 10);
        const firePerimeter = Math.sqrt(burnedArea * 4 * Math.PI);
        const spreadRate = timeInHours > 0 ? burnedArea / timeInHours : 0;
        const activeCount = Math.min(Math.floor(burnedArea / 50) + 1, 15);

        updateElementText('totalBurnedArea', Math.max(0, burnedArea).toFixed(0) + ' ha');
        updateElementText('firePerimeter', Math.max(0, firePerimeter).toFixed(1) + ' km');
        updateElementText('spreadRate', Math.max(0, spreadRate).toFixed(1) + ' ha/hr');
        updateElementText('activeFireSources', activeCount);

        updateSimulationMonitoringChart(burnedArea, firePerimeter);
    } else {
        updateElementText('totalBurnedArea', '0 ha');
        updateElementText('firePerimeter', '0 km');
        updateElementText('spreadRate', '0 ha/hr');
        updateElementText('activeFireSources', '0');

        if (window.chartInstances && window.chartInstances.simulationMonitoring) {
            const chart = window.chartInstances.simulationMonitoring;
            chart.data.labels = ['0m'];
            chart.data.datasets[0].data = [0];
            chart.data.datasets[1].data = [0];
            chart.update('none');
        }
    }
}

function updateSimulationMonitoringChart(burnedArea, firePerimeter) {
    if (window.chartInstances && window.chartInstances.simulationMonitoring) {
        const chart = window.chartInstances.simulationMonitoring;

        const timeLabel = simulationTime > 60 ? 
            Math.floor(simulationTime / 60) + 'h' + (simulationTime % 60 > 0 ? (simulationTime % 60) + 'm' : '') :
            simulationTime + 'm';

        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
            chart.data.datasets[1].data.shift();
        }

        chart.data.labels.push(timeLabel);
        chart.data.datasets[0].data.push(Math.max(0, burnedArea));
        chart.data.datasets[1].data.push(Math.max(0, firePerimeter));

        chart.update('none');
    }
}

// Toast Notifications
function showToast(message, type = 'success', duration = 3000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideInRight 0.3s ease-out reverse';
        setTimeout(() => {
            if (container.contains(toast)) {
                container.removeChild(toast);
            }
        }, 300);
    }, duration);
}

// Modal Functions
function openModal() {
    const overlay = document.getElementById('modal-overlay');
    if (overlay) {
        overlay.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
}

function closeModal() {
    const overlay = document.getElementById('modal-overlay');
    if (overlay) {
        overlay.classList.remove('active');
        document.body.style.overflow = 'auto';
    }
}

// Close modal on overlay click
const modalOverlay = document.getElementById('modal-overlay');
if (modalOverlay) {
    modalOverlay.addEventListener('click', function(e) {
        if (e.target === this) {
            closeModal();
        }
    });
}

// Scroll to section function
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth' });
        showToast(`Navigating to ${sectionId.replace('-', ' ')} section`, 'processing', 1500);
    }
}

// Download report function
function downloadReport() {
    showToast('Generating daily risk report...', 'processing', 2000);

    setTimeout(() => {
        showToast('Daily risk report downloaded successfully!', 'success');

        const link = document.createElement('a');
        link.href = 'data:text/plain;charset=utf-8,NeuroNix Daily Fire Risk Report\n\nGenerated: ' + new Date().toLocaleString() + '\n\nOverall Risk Level: High\nTotal Monitored Area: 53,483 km²\nActive Sensors: 247\nPrediction Accuracy: 97.2%';
        link.download = 'neuronix-daily-report-' + new Date().toISOString().split('T')[0] + '.txt';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }, 2000);
}

// Resource Optimization Functions
function initializeResourceOptimization() {
    const optimizeBtn = document.getElementById('optimize-deployment');
    if (optimizeBtn) {
        optimizeBtn.addEventListener('click', runResourceOptimization);
    }
}

async function runResourceOptimization() {
    const optimizeBtn = document.getElementById('optimize-deployment');
    if (optimizeBtn) {
        optimizeBtn.disabled = true;
        optimizeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Optimizing...';
    }

    try {
        // Get current resource counts
        const firefighters = parseInt(document.getElementById('firefighters-count').value) || 0;
        const waterTanks = parseInt(document.getElementById('water-tanks-count').value) || 0;
        const drones = parseInt(document.getElementById('drones-count').value) || 0;
        const helicopters = parseInt(document.getElementById('helicopters-count').value) || 0;

        // Get current environmental data
        const envData = getCurrentEnvironmentalData();

        const requestData = {
            ...envData,
            firefighters,
            water_tanks: waterTanks,
            drones,
            helicopters
        };

        showToast('Running AI optimization algorithm...', 'processing', 3000);

        // Call ML API for optimization
        const response = await fetch(`${ML_API_BASE}/api/ml/optimize-resources`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                currentOptimization = result.optimization;
                displayOptimizationResults(result.optimization);
                updateDeploymentMap(result.optimization.deployment_plan);
                showToast('Resource optimization completed successfully!', 'success');
            }
        } else {
            throw new Error('API request failed');
        }
    } catch (error) {
        console.warn('ML API unavailable, using fallback optimization');
        const fallbackOptimization = generateFallbackOptimization();
        currentOptimization = fallbackOptimization;
        displayOptimizationResults(fallbackOptimization);
        updateDeploymentMap(fallbackOptimization.deployment_plan);
        showToast('Optimization completed with local algorithm', 'warning');
    }

    if (optimizeBtn) {
        optimizeBtn.disabled = false;
        optimizeBtn.innerHTML = '<i class="fas fa-magic"></i> Optimize Deployment';
    }
}

function displayOptimizationResults(optimization) {
    // Update optimization score
    const scoreElement = document.getElementById('optimization-score');
    if (scoreElement) {
        scoreElement.textContent = optimization.optimization_score + '/100';
    }

    // Update coverage metrics
    updateElementText('overall-coverage', optimization.coverage_metrics.overall_coverage_percentage.toFixed(1) + '%');
    updateElementText('avg-response-time', optimization.response_times.overall.average_minutes.toFixed(1) + ' min');
    updateElementText('districts-covered', optimization.coverage_metrics.total_districts_covered + '/13');

    // Calculate high risk coverage
    const highRiskCoverage = calculateHighRiskCoverage(optimization.deployment_plan);
    updateElementText('high-risk-coverage', highRiskCoverage + '%');

    // Update resource breakdown
    updateResourceBreakdown(optimization.deployment_plan);

    // Update recommendations
    updateRecommendations(optimization.recommendations);
}

function calculateHighRiskCoverage(deploymentPlan) {
    const highRiskDistricts = ['Nainital', 'Almora', 'Chamoli'];
    let covered = 0;

    Object.values(deploymentPlan).forEach(deployments => {
        deployments.forEach(deployment => {
            if (highRiskDistricts.includes(deployment.district)) {
                covered++;
            }
        });
    });

    return Math.min(100, Math.round((covered / highRiskDistricts.length) * 100));
}

function updateResourceBreakdown(deploymentPlan) {
    const container = document.getElementById('resource-breakdown');
    if (!container) return;

    container.innerHTML = '';

    const resourceIcons = {
        firefighters: 'fas fa-users',
        water_tanks: 'fas fa-tint',
        drones: 'fas fa-helicopter',
        helicopters: 'fas fa-plane'
    };

    Object.entries(deploymentPlan).forEach(([resourceType, deployments]) => {
        if (deployments.length > 0) {
            const totalUnits = deployments.reduce((sum, d) => sum + d.units, 0);
            const districtsCount = deployments.length;

            const item = document.createElement('div');
            item.className = 'resource-breakdown-item';
            item.innerHTML = `
                <div class="breakdown-resource">
                    <i class="${resourceIcons[resourceType]}"></i>
                    ${resourceType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </div>
                <div class="breakdown-count">${totalUnits} units in ${districtsCount} districts</div>
            `;
            container.appendChild(item);
        }
    });
}

function updateRecommendations(recommendations) {
    const container = document.getElementById('recommendations-list');
    if (!container) return;

    container.innerHTML = '';

    recommendations.forEach(rec => {
        const item = document.createElement('div');
        item.className = 'recommendation-item';
        item.innerHTML = `
            <i class="fas fa-lightbulb"></i>
            <span>${rec}</span>
        `;
        container.appendChild(item);
    });
}

function updateDeploymentMap(deploymentPlan) {
    if (!deploymentMap) return;

    // Clear existing deployment layers
    deploymentLayers.forEach(layer => {
        try {
            deploymentMap.removeLayer(layer);
        } catch (e) {
            // Ignore errors for layers already removed
        }
    });
    deploymentLayers = [];

    const resourceColors = {
        firefighters: '#FF4500',
        water_tanks: '#3B82F6',
        drones: '#10B981',
        helicopters: '#8B5CF6'
    };

    const resourceIcons = {
        firefighters: 'fas fa-users',
        water_tanks: 'fas fa-tint',
        drones: 'fas fa-helicopter',
        helicopters: 'fas fa-plane'
    };

    Object.entries(deploymentPlan).forEach(([resourceType, deployments]) => {
        deployments.forEach(deployment => {
            const marker = L.marker(deployment.coordinates, {
                icon: L.divIcon({
                    className: 'deployment-marker',
                    html: `
                        <div class="deployment-icon" style="background-color: ${resourceColors[resourceType]};">
                            <i class="${resourceIcons[resourceType]}"></i>
                            <span class="unit-count">${deployment.units}</span>
                        </div>
                    `,
                    iconSize: [40, 40],
                    iconAnchor: [20, 20]
                })
            }).addTo(deploymentMap);

            // Add coverage circle
            const coverageCircle = L.circle(deployment.coordinates, {
                color: resourceColors[resourceType],
                fillColor: resourceColors[resourceType],
                fillOpacity: 0.1,
                radius: deployment.coverage_radius * 1000, // Convert km to meters
                weight: 1
            }).addTo(deploymentMap);

            marker.bindPopup(`
                <div>
                    <h4>${deployment.district}</h4>
                    <p><strong>Resource:</strong> ${resourceType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</p>
                    <p><strong>Units:</strong> ${deployment.units}</p>
                    <p><strong>Risk Score:</strong> ${(deployment.risk_score * 100).toFixed(1)}%</p>
                    <p><strong>Coverage:</strong> ${deployment.coverage_radius} km radius</p>
                </div>
            `);

            deploymentLayers.push(marker);
            deploymentLayers.push(coverageCircle);
        });
    });
}

function addDistrictBoundaries() {
    if (!deploymentMap) return;

    const districts = [
        {
            name: 'Nainital',
            coords: [[29.2, 79.3], [29.6, 79.3], [29.6, 79.8], [29.2, 79.8]],
            risk: 'very-high'
        },
        {
            name: 'Almora',
            coords: [[29.5, 79.5], [29.9, 79.5], [29.9, 80.0], [29.5, 80.0]],
            risk: 'high'
        },
        {
            name: 'Dehradun',
            coords: [[30.1, 77.8], [30.5, 77.8], [30.5, 78.3], [30.1, 78.3]],
            risk: 'moderate'
        }
    ];

    districts.forEach(district => {
        const riskColors = {
            'very-high': '#FF4444',
            'high': '#FFA726',
            'moderate': '#66BB6A',
            'low': '#42A5F5'
        };

        const polygon = L.polygon(district.coords, {
            color: riskColors[district.risk],
            fillColor: riskColors[district.risk],
            fillOpacity: 0.2,
            weight: 2
        }).addTo(deploymentMap);

        polygon.bindTooltip(district.name, { permanent: false, direction: 'center' });
    });
}

function generateFallbackOptimization() {
    // Fallback optimization when ML API is unavailable
    return {
        deployment_plan: {
            firefighters: [
                { district: 'Nainital', coordinates: [29.3806, 79.4422], units: 3, risk_score: 0.85, coverage_radius: 5 },
                { district: 'Almora', coordinates: [29.6500, 79.6667], units: 2, risk_score: 0.68, coverage_radius: 5 }
            ],
            water_tanks: [
                { district: 'Nainital', coordinates: [29.3806, 79.4422], units: 2, risk_score: 0.85, coverage_radius: 3 },
                { district: 'Dehradun', coordinates: [30.3165, 78.0322], units: 1, risk_score: 0.42, coverage_radius: 3 }
            ],
            drones: [
                { district: 'Nainital', coordinates: [29.3806, 79.4422], units: 1, risk_score: 0.85, coverage_radius: 15 },
                { district: 'Almora', coordinates: [29.6500, 79.6667], units: 1, risk_score: 0.68, coverage_radius: 15 },
                { district: 'Chamoli', coordinates: [30.4000, 79.3200], units: 1, risk_score: 0.72, coverage_radius: 15 }
            ],
            helicopters: [
                { district: 'Nainital', coordinates: [29.3806, 79.4422], units: 1, risk_score: 0.85, coverage_radius: 50 }
            ]
        },
        coverage_metrics: {
            overall_coverage_percentage: 78.5,
            total_districts_covered: 10,
            coverage_by_resource: {}
        },
        response_times: {
            overall: {
                average_minutes: 12.3,
                efficiency_score: 87.7
            }
        },
        optimization_score: 82.1,
        recommendations: [
            "Deploy additional helicopters for rapid response in high-risk areas",
            "Increase drone surveillance for better real-time monitoring",
            "Consider mobile water tank units for flexible deployment",
            "Maintain current deployment strategy for optimal coverage"
        ]
    };
}

// Environmental Impact Functions
function initializeEnvironmentalImpact() {
    // Initialize visualization controls
    const vizButtons = document.querySelectorAll('.viz-btn');
    vizButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const vizType = e.target.getAttribute('data-viz');
            switchVisualization(vizType);
        });
    });

    // Initialize analysis buttons
    const carbonAnalysisBtn = document.getElementById('run-carbon-analysis');
    const impactAnalysisBtn = document.getElementById('run-impact-analysis');
    const exportBtn = document.getElementById('export-analysis');

    if (carbonAnalysisBtn) {
        carbonAnalysisBtn.addEventListener('click', runCarbonAnalysis);
    }

    if (impactAnalysisBtn) {
        impactAnalysisBtn.addEventListener('click', runEnvironmentalImpactAnalysis);
    }

    if (exportBtn) {
        exportBtn.addEventListener('click', exportAnalysisReport);
    }

    // Initialize charts
    initializeEnvironmentalCharts();

    showToast('Environmental impact analysis ready', 'success');
}

function initializeEnvironmentalCharts() {
    // Carbon Emissions Chart
    const carbonCtx = document.getElementById('carbonEmissionsChart');
    if (carbonCtx) {
        window.carbonEmissionsChart = new Chart(carbonCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['0h', '1h', '2h', '3h', '4h', '5h', '6h'],
                datasets: [{
                    label: 'CO₂ Emissions (tonnes/hour)',
                    data: [0, 0, 0, 0, 0, 0, 0],
                    borderColor: '#F59E0B',
                    backgroundColor: 'rgba(245, 158, 11, 0.2)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: getChartOptions('CO₂ Emissions Over Time', 'tonnes/hour')
        });
    }

    // Recovery Progress Chart
    const recoveryCtx = document.getElementById('recoveryProgressChart');
    if (recoveryCtx) {
        window.recoveryProgressChart = new Chart(recoveryCtx.getContext('2d'), {
            type: 'radar',
            data: {
                labels: ['Vegetation', 'Soil Health', 'Wildlife', 'Water Cycle', 'Carbon Storage'],
                datasets: [{
                    label: 'Recovery Progress (%)',
                    data: [0, 0, 0, 0, 0],
                    borderColor: '#10B981',
                    backgroundColor: 'rgba(16, 185, 129, 0.2)',
                    pointBackgroundColor: '#10B981'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#ffffff' } }
                },
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#ffffff' },
                        ticks: { color: '#ffffff', backdropColor: 'transparent' },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    // Economic Impact Chart
    const economicCtx = document.getElementById('economicImpactChart');
    if (economicCtx) {
        window.economicImpactChart = new Chart(economicCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Direct Timber Loss', 'Ecosystem Services', 'Recreation/Tourism', 'Restoration Cost'],
                datasets: [{
                    data: [0, 0, 0, 0],
                    backgroundColor: ['#EF4444', '#F59E0B', '#3B82F6', '#10B981'],
                    borderWidth: 2,
                    borderColor: 'rgba(255, 255, 255, 0.1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#ffffff', padding: 15 }
                    }
                }
            }
        });
    }

    // Carbon Loss Chart
    const carbonLossCtx = document.getElementById('carbonLossChart');
    if (carbonLossCtx) {
        window.carbonLossChart = new Chart(carbonLossCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['Year 1-5', 'Year 6-10', 'Year 11-20', 'Year 21-30', 'Year 31+'],
                datasets: [{
                    label: 'Lost Sequestration (tonnes CO₂)',
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: 'rgba(239, 68, 68, 0.6)',
                    borderColor: '#EF4444',
                    borderWidth: 1
                }]
            },
            options: getChartOptions('Carbon Sequestration Loss Over Time', 'tonnes CO₂')
        });
    }
}

function getChartOptions(title, yAxisLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            title: {
                display: true,
                text: title,
                color: '#ffffff',
                font: { size: 14 }
            },
            legend: { labels: { color: '#ffffff' } }
        },
        scales: {
            x: {
                ticks: { color: '#ffffff' },
                grid: { color: 'rgba(255, 255, 255, 0.1)' }
            },
            y: {
                ticks: { color: '#ffffff' },
                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                title: {
                    display: true,
                    text: yAxisLabel,
                    color: '#ffffff'
                }
            }
        }
    };
}

function switchVisualization(vizType) {
    // Update button states
    document.querySelectorAll('.viz-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-viz="${vizType}"]`).classList.add('active');

    // Switch visualization panels
    document.querySelectorAll('.viz-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    document.getElementById(`${vizType}-viz`).classList.add('active');
}

async function runCarbonAnalysis() {
    const analysisBtn = document.getElementById('run-carbon-analysis');
    const originalText = analysisBtn.innerHTML;
    analysisBtn.disabled = true;
    analysisBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Calculating...';

    try {
        const area = parseFloat(document.getElementById('analysis-area').value) || 100;
        const vegetationType = document.getElementById('vegetation-type').value;
        const severity = document.getElementById('fire-severity').value;

        const response = await fetch(`${ML_API_BASE}/api/ml/carbon-emissions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                burned_area_hectares: area,
                vegetation_type: vegetationType,
                fire_intensity: severity + '_intensity'
            })
        });

        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                updateCarbonEmissionsDisplay(result.emissions);
                showToast('Carbon emissions analysis completed', 'success');
            }
        } else {
            throw new Error('API request failed');
        }
    } catch (error) {
        console.warn('Using fallback carbon calculation');
        const fallbackEmissions = generateFallbackCarbonData();
        updateCarbonEmissionsDisplay(fallbackEmissions);
        showToast('Carbon analysis completed with local data', 'warning');
    }

    analysisBtn.disabled = false;
    analysisBtn.innerHTML = originalText;
}

async function runEnvironmentalImpactAnalysis() {
    const analysisBtn = document.getElementById('run-impact-analysis');
    const originalText = analysisBtn.innerHTML;
    analysisBtn.disabled = true;
    analysisBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

    try {
        const area = parseFloat(document.getElementById('analysis-area').value) || 100;
        const vegetationType = document.getElementById('vegetation-type').value;
        const severity = document.getElementById('fire-severity').value;

        const response = await fetch(`${ML_API_BASE}/api/ml/environmental-impact`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                burned_area_hectares: area,
                vegetation_type: vegetationType,
                fire_severity: severity
            })
        });

        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                updateEnvironmentalImpactDisplay(result.impact);
                showToast('Environmental impact analysis completed', 'success');
            }
        } else {
            throw new Error('API request failed');
        }
    } catch (error) {
        console.warn('Using fallback environmental impact calculation');
        const fallbackImpact = generateFallbackEnvironmentalData();
        updateEnvironmentalImpactDisplay(fallbackImpact);
        showToast('Impact analysis completed with local data', 'warning');
    }

    analysisBtn.disabled = false;
    analysisBtn.innerHTML = originalText;
}

function updateCarbonEmissionsDisplay(emissions) {
    // Update summary metrics
    updateElementText('total-co2-emissions', emissions.total_co2_emissions_tonnes.toFixed(2));
    updateElementText('car-equivalent', Math.round(emissions.equivalent_metrics.car_driving_km).toLocaleString());
    updateElementText('tree-equivalent', Math.round(emissions.equivalent_metrics.trees_annual_absorption));
    updateElementText('household-equivalent', emissions.equivalent_metrics.households_annual_emissions.toFixed(1));

    // Update emissions chart with simulation data
    if (window.carbonEmissionsChart && isSimulationRunning) {
        const hours = Math.min(simulationTime / 10, 6); // Scale simulation time to hours
        const hourlyRate = emissions.total_co2_emissions_tonnes / Math.max(hours, 1);

        const newData = Array.from({length: 7}, (_, i) => 
            i <= hours ? hourlyRate * (i + 1) * 0.8 : 0
        );

        window.carbonEmissionsChart.data.datasets[0].data = newData;
        window.carbonEmissionsChart.update('none');
    }
}

function updateEnvironmentalImpactDisplay(impact) {
    // Update severity score
    updateElementText('severity-score', impact.overall_severity_score.toFixed(1));

    // Update impact categories
    updateElementText('biodiversity-impact', `${impact.biodiversity_impact.estimated_species_loss_percent.toFixed(1)}% loss`);
    updateElementText('soil-impact', impact.soil_impact.soil_erosion_risk_level);
    updateElementText('water-impact', impact.water_cycle_impact.flood_risk_increase);
    updateElementText('economic-impact', `$${(impact.economic_impact_usd.total_economic_impact_usd / 1000000).toFixed(2)}M`);

    // Update recovery timeline
    if (impact.recovery_timeline_years) {
        updateElementText('early-recovery-duration', `${impact.recovery_timeline_years.vegetation_regrowth} years`);
        updateElementText('restoration-duration', `${impact.recovery_timeline_years.wildlife_habitat} years`);
        updateElementText('full-recovery-duration', `${impact.recovery_timeline_years.full_canopy_recovery} years`);
    }

    // Update key statistics
    updateElementText('carbon-sequestration-loss', `${impact.carbon_sequestration_loss.total_sequestration_loss_tonnes_co2.toFixed(0)} tonnes`);
    updateElementText('total-economic-impact', `$${(impact.economic_impact_usd.total_economic_impact_usd / 1000000).toFixed(2)} million`);
    updateElementText('restoration-cost', `$${(impact.economic_impact_usd.recovery_cost_estimate_usd / 1000).toFixed(0)} thousand`);
    updateElementText('species-loss-risk', `${impact.biodiversity_impact.estimated_species_loss_percent.toFixed(1)}%`);

    // Update recommendations
    updateRecommendationsList(impact.mitigation_recommendations);

    // Update charts
    updateEnvironmentalCharts(impact);
}

function updateRecommendationsList(recommendations) {
    const container = document.getElementById('environmental-recommendations');
    if (!container) return;

    container.innerHTML = '';

    recommendations.forEach(recommendation => {
        const item = document.createElement('div');
        item.className = 'recommendation-item';
        item.innerHTML = `
            <i class="fas fa-leaf"></i>
            <span>${recommendation}</span>
        `;
        container.appendChild(item);
    });
}

function updateEnvironmentalCharts(impact) {
    // Update recovery progress chart
    if (window.recoveryProgressChart) {
        const recoveryData = [
            Math.max(0, 100 - impact.biodiversity_impact.estimated_species_loss_percent),
            impact.soil_impact.soil_erosion_risk_level === 'low' ? 80 : 
            impact.soil_impact.soil_erosion_risk_level === 'moderate' ? 60 : 30,
            Math.max(0, 100 - impact.biodiversity_impact.estimated_species_loss_percent * 0.8),
            impact.water_cycle_impact.flood_risk_increase === 'low' ? 70 : 40,
            Math.max(0, 100 - impact.carbon_sequestration_loss.total_sequestration_loss_tonnes_co2 / 1000)
        ];

        window.recoveryProgressChart.data.datasets[0].data = recoveryData;
        window.recoveryProgressChart.update('none');
    }

    // Update economic impact chart
    if (window.economicImpactChart) {
        const economicData = [
            impact.economic_impact_usd.direct_timber_loss_usd,
            impact.economic_impact_usd.ecosystem_service_loss_usd,
            impact.economic_impact_usd.recreation_tourism_loss_usd,
            impact.economic_impact_usd.recovery_cost_estimate_usd
        ];

        window.economicImpactChart.data.datasets[0].data = economicData;
        window.economicImpactChart.update('none');
    }

    // Update carbon loss chart
    if (window.carbonLossChart) {
        const totalLoss = impact.carbon_sequestration_loss.total_sequestration_loss_tonnes_co2;
        const lossData = [
            totalLoss * 0.3,  // Years 1-5
            totalLoss * 0.25, // Years 6-10
            totalLoss * 0.2,  // Years 11-20
            totalLoss * 0.15, // Years 21-30
            totalLoss * 0.1   // Years 31+
        ];

        window.carbonLossChart.data.datasets[0].data = lossData;
        window.carbonLossChart.update('none');
    }
}

function generateFallbackCarbonData() {
    const area = parseFloat(document.getElementById('analysis-area').value) || 100;
    const vegetationType = document.getElementById('vegetation-type').value;

    const emissionFactors = {
        'coniferous': 1.83,
        'deciduous': 1.79,
        'mixed_forest': 1.81,
        'grassland': 1.76,
        'shrubland': 1.78
    };

    const biomassDensity = {
        'coniferous': 45,
        'deciduous': 35,
        'mixed_forest': 40,
        'grassland': 2.5,
        'shrubland': 8
    };

    const factor = emissionFactors[vegetationType] || 1.81;
    const density = biomassDensity[vegetationType] || 40;
    const burnedBiomass = area * 10000 * density * 0.35; // 35% combustion
    const totalEmissions = burnedBiomass * factor;

    return {
        total_co2_emissions_kg: totalEmissions,
        total_co2_emissions_tonnes: totalEmissions / 1000,
        burned_area_hectares: area,
        vegetation_type: vegetationType,
        equivalent_metrics: {
            car_driving_km: totalEmissions / 0.12,
            trees_annual_absorption: totalEmissions / 22,
            households_annual_emissions: (totalEmissions / 1000) / 4.6
        }
    };
}

function generateFallbackEnvironmentalData() {
    const area = parseFloat(document.getElementById('analysis-area').value) || 100;
    const severity = document.getElementById('fire-severity').value;

    const severityMultipliers = {
        'low': 0.7, 'moderate': 1.0, 'high': 1.5, 'severe': 2.2, 'extreme': 3.0
    };

    const multiplier = severityMultipliers[severity] || 1.0;

    return {
        overall_severity_score: Math.min(10, 4 * multiplier),
        biodiversity_impact: {
            estimated_species_loss_percent: Math.min(95, 25 * multiplier)
        },
        soil_impact: {
            soil_erosion_risk_level: multiplier > 1.5 ? 'high' : multiplier > 1.0 ? 'moderate' : 'low'
        },
        water_cycle_impact: {
            flood_risk_increase: multiplier > 1.5 ? 'high' : 'moderate'
        },
        economic_impact_usd: {
            total_economic_impact_usd: area * 5000 * multiplier,
            direct_timber_loss_usd: area * 400 * multiplier,
            ecosystem_service_loss_usd: area * 3000 * multiplier,
            recreation_tourism_loss_usd: area * 1200 * multiplier,
            recovery_cost_estimate_usd: area * 2500
        },
        carbon_sequestration_loss: {
            total_sequestration_loss_tonnes_co2: area * 7.3 * 25 * multiplier
        },
        recovery_timeline_years: {
            vegetation_regrowth: Math.round(12 * multiplier),
            wildlife_habitat: Math.round(16 * multiplier),
            full_canopy_recovery: Math.round(32 * multiplier)
        },
        mitigation_recommendations: [
            "Implement immediate erosion control measures",
            "Conduct comprehensive soil testing",
            "Priority reforestation with native species",
            "Establish wildlife corridors",
            "Monitor water quality downstream"
        ]
    };
}

function exportAnalysisReport() {
    showToast('Generating environmental analysis report...', 'processing', 2000);

    const area = document.getElementById('analysis-area').value;
    const vegetationType = document.getElementById('vegetation-type').value;
    const severity = document.getElementById('fire-severity').value;

    const reportContent = `
NeuroNix Environmental Impact Analysis Report
Generated: ${new Date().toLocaleString()}

Analysis Parameters:
- Burned Area: ${area} hectares
- Vegetation Type: ${vegetationType}
- Fire Severity: ${severity}

Carbon Emissions Summary:
- Total CO₂ Emissions: ${document.getElementById('total-co2-emissions').textContent} tonnes
- Car Driving Equivalent: ${document.getElementById('car-equivalent').textContent} km
- Tree Absorption Equivalent: ${document.getElementById('tree-equivalent').textContent} trees/year

Environmental Impact:
- Overall Severity Score: ${document.getElementById('severity-score').textContent}/10
- Species Loss Risk: ${document.getElementById('species-loss-risk').textContent}
- Carbon Sequestration Loss: ${document.getElementById('carbon-sequestration-loss').textContent}
- Total Economic Impact: ${document.getElementById('total-economic-impact').textContent}

Recovery Timeline:
- Early Recovery: ${document.getElementById('early-recovery-duration').textContent}
- Ecosystem Restoration: ${document.getElementById('restoration-duration').textContent}
- Full Recovery: ${document.getElementById('full-recovery-duration').textContent}

Generated by NeuroNix Forest Fire Intelligence Platform
    `.trim();

    setTimeout(() => {
        const link = document.createElement('a');
        link.href = 'data:text/plain;charset=utf-8,' + encodeURIComponent(reportContent);
        link.download = `neuronix-environmental-impact-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        showToast('Environmental analysis report downloaded successfully!', 'success');
    }, 2000);
}

// Enhanced keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('.search-input');
        if (searchInput) {
            searchInput.focus();
            showToast('Search activated', 'processing', 1000);
        }
    }

    if (e.key === 'Escape') {
        closeModal();
    }

    if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault();
        downloadReport();
    }

    if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
        e.preventDefault();
        const section = document.getElementById('resource-optimization');
        if (section) {
            section.scrollIntoView({ behavior: 'smooth' });
            showToast('Resource Optimization activated', 'processing', 1500);
        }
    }
});

// Training Mode Implementation
class TrainingMode {
    constructor() {
        this.trainingMap = null;
        this.currentScenario = 'forest-fire';
        this.selectedAction = null;
        this.isTrainingActive = false;
        this.trainingScore = 0;
        this.trainingLevel = 1;
        this.scenariosCompleted = 0;
        this.bestScore = 0;
        this.startTime = null;
        this.timerInterval = null;
        this.fireSimulation = {
            withActions: [],
            withoutActions: []
        };
        this.placedActions = [];
        this.achievements = {
            'first-save': false,
            'fire-expert': false,
            'community-hero': false,
            'perfect-score': false
        };

        this.init();
    }

    init() {
        this.initializeTrainingMap();
        this.setupEventListeners();
        this.updateDisplay();
        this.initializeChart();
        this.loadProgress();
    }

    initializeTrainingMap() {
        const mapElement = document.getElementById('training-map');
        if (!mapElement || typeof L === 'undefined') return;

        this.trainingMap = L.map('training-map').setView([30.0668, 79.0193], 12);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.trainingMap);

        // Add forest area for training
        const forestArea = L.polygon([
            [30.05, 79.00],
            [30.08, 79.00],
            [30.08, 79.03],
            [30.05, 79.03]
        ], {
            color: '#2d5a2d',
            fillColor: '#2d5a2d',
            fillOpacity: 0.6
        }).addTo(this.trainingMap);

        forestArea.bindPopup('<h4>Training Forest Area</h4><p>Click here to start a fire simulation</p>');

        // Map click handler for placing actions
        this.trainingMap.on('click', (e) => {
            if (this.selectedAction && this.isTrainingActive) {
                this.placeAction(e.latlng, this.selectedAction);
            } else if (!this.isTrainingActive) {
                this.startFireAt(e.latlng);
            }
        });
    }

    setupEventListeners() {
        // Scenario selection
        document.querySelectorAll('.scenario-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
                e.target.closest('.scenario-btn').classList.add('active');
                this.currentScenario = e.target.closest('.scenario-btn').getAttribute('data-scenario');
                this.resetScenario();
            });
        });

        // Action selection
        document.querySelectorAll('.action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.action-btn').forEach(b => b.classList.remove('selected'));
                e.target.closest('.action-btn').classList.add('selected');
                this.selectedAction = e.target.closest('.action-btn').getAttribute('data-action');
                this.updateInstructions();
            });
        });

        // Control buttons
        document.getElementById('start-training')?.addEventListener('click', () => this.startTraining());
        document.getElementById('pause-training')?.addEventListener('click', () => this.pauseTraining());
        document.getElementById('reset-training')?.addEventListener('click', () => this.resetTraining());

        // Comparison toggle
        document.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.updateComparison(e.target.getAttribute('data-view'));
            });
        });
    }

    startTraining() {
        this.isTrainingActive = true;
        this.startTime = Date.now();
        this.placedActions = [];

        // Start timer
        this.timerInterval = setInterval(() => {
            this.updateTimer();
        }, 1000);

        // Start fire simulation without actions
        this.runSimulationWithoutActions();

        // Update UI
        document.getElementById('training-status').textContent = 'Active';
        document.getElementById('training-status').style.background = 'rgba(245, 158, 11, 0.2)';
        document.getElementById('training-status').style.color = '#F59E0B';
        document.getElementById('training-status').style.borderColor = 'rgba(245, 158, 11, 0.3)';

        this.updateInstructions('Select an action and click on the map to place it!');

        showToast('Training scenario started! Click on the map to place fire control actions.', 'success');
    }

    pauseTraining() {
        this.isTrainingActive = false;
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
        }

        document.getElementById('training-status').textContent = 'Paused';
        document.getElementById('training-status').style.background = 'rgba(239, 68, 68, 0.2)';
        document.getElementById('training-status').style.color = '#EF4444';
        document.getElementById('training-status').style.borderColor = 'rgba(239, 68, 68, 0.3)';
    }

    resetTraining() {
        this.isTrainingActive = false;
        this.selectedAction = null;
        this.placedActions = [];
        this.fireSimulation = { withActions: [], withoutActions: [] };

        if (this.timerInterval) {
            clearInterval(this.timerInterval);
        }

        // Clear map markers
        this.trainingMap.eachLayer(layer => {
            if (layer instanceof L.Marker || (layer instanceof L.Circle && layer.options.className === 'training-fire')) {
                this.trainingMap.removeLayer(layer);
            }
        });

        // Reset UI
        document.getElementById('training-status').textContent = 'Ready';
        document.getElementById('training-status').style.background = 'rgba(16, 185, 129, 0.2)';
        document.getElementById('training-status').style.color = '#10B981';
        document.getElementById('training-status').style.borderColor = 'rgba(16, 185, 129, 0.3)';

        document.getElementById('training-timer').textContent = '00:00';

        document.querySelectorAll('.action-btn').forEach(btn => btn.classList.remove('selected'));

        this.updateInstructions('Click on the map to start a new fire simulation!');
        this.resetMetrics();
    }

    startFireAt(latlng) {
        if (this.isTrainingActive) return;

        // Add fire marker
        const fireMarker = L.marker([latlng.lat, latlng.lng], {
            icon: L.divIcon({
                className: 'fire-marker training-fire',
                html: '<i class="fas fa-fire" style="color: #EF4444; font-size: 24px;"></i>',
                iconSize: [30, 30],
                iconAnchor: [15, 15]
            })
        }).addTo(this.trainingMap);

        // Start training automatically
        setTimeout(() => {
            this.startTraining();
        }, 500);

        this.updateInstructions('Fire started! Select actions to control the fire spread.');
    }

    placeAction(latlng, actionType) {
        if (!this.isTrainingActive) return;

        const actionCosts = {
            'fireline': 10,
            'water': 15,
            'evacuate': 5,
            'burn': 20
        };

        const cost = actionCosts[actionType] || 10;

        if (this.trainingScore < cost) {
            showToast(`Insufficient points! Need ${cost} points for this action.`, 'warning');
            return;
        }

        // Deduct cost
        this.trainingScore -= cost;
        this.updateDisplay();

        // Add action marker
        const marker = L.marker([latlng.lat, latlng.lng], {
            icon: L.divIcon({
                className: `training-action-marker ${actionType}`,
                html: this.getActionIcon(actionType),
                iconSize: [20, 20],
                iconAnchor: [10, 10]
            })
        }).addTo(this.trainingMap);

        // Store action
        this.placedActions.push({
            type: actionType,
            latlng: latlng,
            marker: marker,
            effectiveness: this.calculateActionEffectiveness(actionType)
        });

        // Calculate impact and award points
        const impact = this.calculateActionImpact(actionType, latlng);
        const points = Math.round(impact * 50);
        this.trainingScore += points;

        this.updateDisplay();
        this.updateMetrics();

        // Show feedback
        const feedback = this.generateActionFeedback(actionType, impact);
        this.showActionFeedback(feedback);

        // Check for scenario completion
        if (this.placedActions.length >= 3) {
            setTimeout(() => {
                this.completeScenario();
            }, 2000);
        }

        // Clear selection
        this.selectedAction = null;
        document.querySelectorAll('.action-btn').forEach(btn => btn.classList.remove('selected'));

        showToast(`${actionType.charAt(0).toUpperCase() + actionType.slice(1)} deployed! +${points} points`, 'success');
    }

    getActionIcon(actionType) {
        const icons = {
            'fireline': '<i class="fas fa-minus" style="color: white; font-size: 12px;"></i>',
            'water': '<i class="fas fa-tint" style="color: white; font-size: 12px;"></i>',
            'evacuate': '<i class="fas fa-users" style="color: white; font-size: 12px;"></i>',
            'burn': '<i class="fas fa-fire" style="color: white; font-size: 12px;"></i>'
        };
        return icons[actionType] || '<i class="fas fa-circle"></i>';
    }

    calculateActionEffectiveness(actionType) {
        const effectiveness = {
            'fireline': 0.8,
            'water': 0.9,
            'evacuate': 0.6,
            'burn': 0.7
        };
        return effectiveness[actionType] || 0.5;
    }

    calculateActionImpact(actionType, latlng) {
        // Simulate impact based on action type and placement
        let baseImpact = 0.5;

        switch(actionType) {
            case 'fireline':
                baseImpact = 0.7;
                break;
            case 'water':
                baseImpact = 0.8;
                break;
            case 'evacuate':
                baseImpact = 0.4;
                break;
            case 'burn':
                baseImpact = 0.6;
                break;
        }

        // Add some randomness for realism
        return baseImpact + (Math.random() - 0.5) * 0.3;
    }

    generateActionFeedback(actionType, impact) {
        const impactPercent = Math.round(impact * 100);

        if (impact > 0.7) {
            return {
                type: 'success',
                message: `Excellent placement! Your ${actionType} action reduced fire spread by ${impactPercent}%.`
            };
        } else if (impact > 0.4) {
            return {
                type: 'warning',
                message: `Good effort! Your ${actionType} action had moderate impact (${impactPercent}% reduction).`
            };
        } else {
            return {
                type: 'info',
                message: `Consider better placement. Your ${actionType} action had limited impact (${impactPercent}% reduction).`
            };
        }
    }

    showActionFeedback(feedback) {
        const feedbackContainer = document.getElementById('feedback-content');
        if (!feedbackContainer) return;

        // Remove placeholder
        const placeholder = feedbackContainer.querySelector('.placeholder');
        if (placeholder) {
            placeholder.remove();
        }

        const feedbackItem = document.createElement('div');
        feedbackItem.className = `feedback-item ${feedback.type}`;
        feedbackItem.innerHTML = `
            <i class="fas fa-${feedback.type === 'success' ? 'check-circle' : feedback.type === 'warning' ? 'exclamation-triangle' : 'info-circle'}"></i>
            <span>${feedback.message}</span>
        `;

        feedbackContainer.appendChild(feedbackItem);

        // Keep only last 3 feedback items
        const items = feedbackContainer.querySelectorAll('.feedback-item');
        if (items.length > 3) {
            items[0].remove();
        }
    }

    completeScenario() {
        this.isTrainingActive = false;
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
        }

        // Calculate final score
        const timeBonus = Math.max(0, 300 - (Date.now() - this.startTime) / 1000) * 2;
        const effectivenessBonus = this.calculateOverallEffectiveness() * 100;
        const finalScore = Math.round(this.trainingScore + timeBonus + effectivenessBonus);

        this.trainingScore = finalScore;
        this.scenariosCompleted++;

        if (finalScore > this.bestScore) {
            this.bestScore = finalScore;
            this.checkAchievements();
        }

        // Level up logic
        if (this.scenariosCompleted % 3 === 0) {
            this.trainingLevel++;
            showToast(`Level up! You are now Level ${this.trainingLevel}`, 'success');
        }

        this.updateDisplay();
        this.saveProgress();

        // Show completion feedback
        this.showCompletionFeedback(finalScore, timeBonus, effectivenessBonus);

        document.getElementById('training-status').textContent = 'Completed';
        document.getElementById('training-status').style.background = 'rgba(16, 185, 129, 0.2)';
        document.getElementById('training-status').style.color = '#10B981';
        document.getElementById('training-status').style.borderColor = 'rgba(16, 185, 129, 0.3)';

        showToast(`Scenario completed! Final score: ${finalScore} points`, 'success');
    }

    calculateOverallEffectiveness() {
        if (this.placedActions.length === 0) return 0;

        const totalEffectiveness = this.placedActions.reduce((sum, action) => {
            return sum + action.effectiveness;
        }, 0);

        return totalEffectiveness / this.placedActions.length;
    }

    showCompletionFeedback(finalScore, timeBonus, effectivenessBonus) {
        const feedbackContainer = document.getElementById('feedback-content');
        if (!feedbackContainer) return;

        feedbackContainer.innerHTML = '';

        const completionFeedback = document.createElement('div');
        completionFeedback.className = 'feedback-item success';
        completionFeedback.innerHTML = `
            <i class="fas fa-trophy"></i>
            <div>
                <strong>Scenario Completed!</strong><br>
                Base Score: ${Math.round(finalScore - timeBonus - effectivenessBonus)}<br>
                Time Bonus: +${Math.round(timeBonus)}<br>
                Effectiveness Bonus: +${Math.round(effectivenessBonus)}<br>
                <strong>Final Score: ${finalScore}</strong>
            </div>
        `;

        feedbackContainer.appendChild(completionFeedback);
    }

    checkAchievements() {
        // First Save achievement
        if (!this.achievements['first-save'] && this.scenariosCompleted >= 1) {
            this.unlockAchievement('first-save');
        }

        // Fire Expert achievement
        if (!this.achievements['fire-expert'] && this.scenariosCompleted >= 5) {
            this.unlockAchievement('fire-expert');
        }

        // Community Hero achievement
        if (!this.achievements['community-hero'] && this.trainingLevel >= 3) {
            this.unlockAchievement('community-hero');
        }

        // Perfect Score achievement
        if (!this.achievements['perfect-score'] && this.bestScore >= 500) {
            this.unlockAchievement('perfect-score');
        }
    }

    unlockAchievement(badgeId) {
        this.achievements[badgeId] = true;

        const badge = document.querySelector(`[data-badge="${badgeId}"]`);
        if (badge) {
            badge.classList.remove('locked');
            badge.classList.add('unlocked');
        }

        showToast(`Achievement unlocked: ${this.getAchievementName(badgeId)}!`, 'success');
    }

    getAchievementName(badgeId) {
        const names = {
            'first-save': 'First Save',
            'fire-expert': 'Fire Expert',
            'community-hero': 'Community Hero',
            'perfect-score': 'Perfect Score'
        };
        return names[badgeId] || badgeId;
    }

    updateTimer() {
        if (!this.startTime) return;

        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;

        document.getElementById('training-timer').textContent = 
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    updateInstructions(message = null) {
        const instructionCard = document.getElementById('instruction-card');
        if (!instructionCard) return;

        if (message) {
            instructionCard.innerHTML = `<i class="fas fa-info-circle"></i><p>${message}</p>`;
            return;
        }

        if (this.selectedAction) {
            const actionNames = {
                'fireline': 'Create a fireline',
                'water': 'Deploy water bombers',
                'evacuate': 'Evacuate civilians',
                'burn': 'Start controlled burn'
            };

            instructionCard.innerHTML = `
                <i class="fas fa-hand-pointer"></i>
                <p>Click on the map to ${actionNames[this.selectedAction] || 'place action'}. Cost: ${this.getActionCost(this.selectedAction)} points</p>
            `;
        } else {
            instructionCard.innerHTML = `
                <i class="fas fa-info-circle"></i>
                <p>Select an action from the left panel, then click on the map to place it.</p>
            `;
        }
    }

    getActionCost(actionType) {
        const costs = { 'fireline': 10, 'water': 15, 'evacuate': 5, 'burn': 20 };
        return costs[actionType] || 10;
    }

    updateDisplay() {
        document.getElementById('training-score').textContent = this.trainingScore;
        document.getElementById('training-level').textContent = this.trainingLevel;
        document.getElementById('scenarios-completed').textContent = this.scenariosCompleted;
        document.getElementById('best-score').textContent = this.bestScore;
    }

    updateMetrics() {
        // Simulate fire reduction based on actions
        const fireReduction = this.calculateFireReduction();
        const structuresSaved = Math.round(fireReduction * 15);
        const forestPreserved = Math.round(fireReduction * 25);
        const responseTime = Math.max(1, 15 - this.placedActions.length * 3);

        document.getElementById('fire-reduction').textContent = `${Math.round(fireReduction)}%`;
        document.getElementById('structures-saved').textContent = structuresSaved;
        document.getElementById('forest-preserved').textContent = forestPreserved;
        document.getElementById('response-time').textContent = responseTime;

        this.updateChart(fireReduction);
    }

    calculateFireReduction() {
        if (this.placedActions.length === 0) return 0;

        let totalReduction = 0;
        this.placedActions.forEach(action => {
            totalReduction += action.effectiveness * 25; // Max 25% per action
        });

        return Math.min(totalReduction, 85); // Cap at 85% reduction
    }

    resetMetrics() {
        document.getElementById('fire-reduction').textContent = '0%';
        document.getElementById('structures-saved').textContent = '0';
        document.getElementById('forest-preserved').textContent = '0';
        document.getElementById('response-time').textContent = '0';

        this.updateChart(0);
    }

    initializeChart() {
        const ctx = document.getElementById('trainingImpactChart');
        if (!ctx || typeof Chart === 'undefined') return;

        this.impactChart = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['Without Actions', 'With Actions'],
                datasets: [{
                    label: 'Area Burned (hectares)',
                    data: [100, 100],
                    backgroundColor: ['rgba(239, 68, 68, 0.6)', 'rgba(16, 185, 129, 0.6)'],
                    borderColor: ['#EF4444', '#10B981'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#ffffff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        ticks: { color: '#ffffff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        title: {
                            display: true,
                            text: 'Area Burned (hectares)',
                            color: '#ffffff'
                        }
                    }
                }
            }
        });
    }

    updateChart(fireReduction) {
        if (!this.impactChart) return;

        const withoutActions = 100;
        const withActions = Math.round(withoutActions * (100 - fireReduction) / 100);

        this.impactChart.data.datasets[0].data = [withoutActions, withActions];
        this.impactChart.update('none');
    }

    updateComparison(view) {
        // Toggle between showing results with and without actions
        if (view === 'without-actions') {
            this.resetMetrics();
        } else {
            this.updateMetrics();
        }
    }

    runSimulationWithoutActions() {
        // Simulate fire spread without any intervention
        this.fireSimulation.withoutActions = Array.from({length: 10}, (_, i) => ({
            hour: i,
            burnedArea: i * i * 2 + Math.random() * 10
        }));
    }

    resetScenario() {
        this.resetTraining();

        // Update scenario-specific elements
        const scenarioDescriptions = {
            'forest-fire': 'Dense forest fire scenario - focus on protecting vegetation and wildlife',
            'grassland-fire': 'Fast-spreading grassland fire - quick response needed',
            'urban-interface': 'Wildland-urban interface fire - protect structures and evacuate civilians'
        };

        this.updateInstructions(scenarioDescriptions[this.currentScenario] || 'Click on the map to start the simulation');
    }

    saveProgress() {
        const progress = {
            trainingLevel: this.trainingLevel,
            scenariosCompleted: this.scenariosCompleted,
            bestScore: this.bestScore,
            achievements: this.achievements
        };

        localStorage.setItem('neuronix-training-progress', JSON.stringify(progress));
    }

    loadProgress() {
        const saved = localStorage.getItem('neuronix-training-progress');
        if (saved) {
            try {
                const progress = JSON.parse(saved);
                this.trainingLevel = progress.trainingLevel || 1;
                this.scenariosCompleted = progress.scenariosCompleted || 0;
                this.bestScore = progress.bestScore || 0;
                this.achievements = { ...this.achievements, ...progress.achievements };

                // Update UI
                this.updateDisplay();

                // Update achievement badges
                Object.keys(this.achievements).forEach(badgeId => {
                    if (this.achievements[badgeId]) {
                        const badge = document.querySelector(`[data-badge="${badgeId}"]`);
                        if (badge) {
                            badge.classList.remove('locked');
                            badge.classList.add('unlocked');
                        }
                    }
                });
            } catch (e) {
                console.warn('Failed to load training progress:', e);
            }
        }
    }
}

// Initialize Training Mode when DOM is loaded
let trainingMode;

document.addEventListener('DOMContentLoaded', function() {
    // ... existing initialization code ...

    // Initialize Training Mode
    setTimeout(() => {
        trainingMode = new TrainingMode();
    }, 1000); // Delay to ensure maps are loaded
});

// Responsive Design for Footer */