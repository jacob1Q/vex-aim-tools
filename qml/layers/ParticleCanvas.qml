import QtQuick 2.15

Canvas {
    id: canvas
    property var particleModel
    property var landmarkModel
    property var particleSummary
    property var viewState
    property color backgroundColor: "#111820"
    property color gridColor: "#1d2a35"
    property int gridSpacingMm: 200

    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()
    onViewStateChanged: requestPaint()

    Connections {
        target: canvas.particleModel
        enabled: canvas.particleModel !== null && canvas.particleModel !== undefined
        function onRevisionChanged() { canvas.requestPaint(); }
    }

    Connections {
        target: canvas.landmarkModel
        enabled: canvas.landmarkModel !== null && canvas.landmarkModel !== undefined
        function onRevisionChanged() { canvas.requestPaint(); }
    }

    Connections {
        target: canvas.particleSummary
        enabled: canvas.particleSummary !== null && canvas.particleSummary !== undefined
        function onChanged() { canvas.requestPaint(); }
    }

    Connections {
        target: canvas.viewState
        enabled: canvas.viewState !== null && canvas.viewState !== undefined
        function onChanged() { canvas.requestPaint(); }
    }

    function pixelsPerMm() {
        if (!viewState || viewState.zoom === undefined || viewState.zoom === null)
            return 0.4;
        return Math.max(0.01, viewState.zoom);
    }

    function centerX() {
        if (!viewState || viewState.centerX === undefined)
            return 0;
        return viewState.centerX;
    }

    function centerY() {
        if (!viewState || viewState.centerY === undefined)
            return 0;
        return viewState.centerY;
    }

    function mapPoint(x, y) {
        var scale = pixelsPerMm();
        return {
            x: (centerY() - y) * scale + width / 2,
            y: (centerX() - x) * scale + height / 2
        };
    }

    function drawGrid(ctx) {
        var scale = pixelsPerMm();
        if (scale <= 0.0)
            return;

        var spacing = Math.max(50, gridSpacingMm);
        var spanX = width / scale;
        var spanY = height / scale;
        var minX = centerX() - spanX / 2;
        var maxX = centerX() + spanX / 2;
        var minY = centerY() - spanY / 2;
        var maxY = centerY() + spanY / 2;
        var startX = Math.floor(minX / spacing) * spacing;
        var startY = Math.floor(minY / spacing) * spacing;

        ctx.save();
        ctx.strokeStyle = gridColor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (var gx = startX; gx <= maxX; gx += spacing) {
            var a = mapPoint(gx, minY);
            var b = mapPoint(gx, maxY);
            ctx.moveTo(a.x + 0.5, a.y + 0.5);
            ctx.lineTo(b.x + 0.5, b.y + 0.5);
        }
        for (var gy = startY; gy <= maxY; gy += spacing) {
            var c = mapPoint(minX, gy);
            var d = mapPoint(maxX, gy);
            ctx.moveTo(c.x + 0.5, c.y + 0.5);
            ctx.lineTo(d.x + 0.5, d.y + 0.5);
        }
        ctx.stroke();
        ctx.restore();
    }

    function drawParticles(ctx) {
        if (!particleModel)
            return;
        var count = particleModel.count || 0;
        var scale = pixelsPerMm();
        var baseSize = 10 * scale;  // Match legacy: height=10mm
        for (var i = 0; i < count; ++i) {
            var entry = particleModel.get(i);
            if (!entry)
                continue;
            var pos = mapPoint(entry.x, entry.y);
            var weight = Math.max(0.0, Math.min(1.0, entry.weight));
            var alpha = 0.2 + weight * 0.6;
            var size = baseSize * (0.8 + weight * 0.4);
            
            // Legacy color: (1, pscale, pscale) where pscale = 1 - weight
            // Red gradient: high weight -> bright red, low weight -> pale red
            var pscale = 1.0 - weight;
            var redG = Math.round(pscale * 255);
            var redB = Math.round(pscale * 255);
            
            ctx.save();
            ctx.translate(pos.x, pos.y);
            ctx.rotate(-(entry.theta + Math.PI / 2));
            ctx.beginPath();
            ctx.moveTo(size, 0);
            ctx.lineTo(-size * 0.6, size * 0.5);
            ctx.lineTo(-size * 0.6, -size * 0.5);
            ctx.closePath();
            ctx.fillStyle = "rgba(255," + redG + "," + redB + "," + alpha + ")";
            ctx.fill();
            ctx.restore();
        }
    }

    function drawSummary(ctx) {
        if (!particleSummary || !particleSummary.isValid)
            return;
        var scale = pixelsPerMm();
        var center = mapPoint(particleSummary.poseX, particleSummary.poseY);
        var major = particleSummary.ellipseMajor * scale;
        var minor = particleSummary.ellipseMinor * scale;
        ctx.save();
        ctx.translate(center.x, center.y);
        ctx.rotate(-(particleSummary.ellipseAngle + Math.PI / 2));
        ctx.beginPath();
        ctx.ellipse(0, 0, major, minor, 0, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(120, 240, 210, 0.6)";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();
    }

    function drawRobot(ctx) {
        // Draw the robot as a large yellow triangle at the best pose estimate
        // Legacy: height=60mm, tip_offset=-10mm, color=(1,1,0,0.7)
        if (!particleSummary || !particleSummary.isValid)
            return;
        var scale = pixelsPerMm();
        var pos = mapPoint(particleSummary.poseX, particleSummary.poseY);
        var size = 60 * scale;  // Match legacy height=60mm
        var tipOffset = -10 * scale;  // Match legacy tip_offset=-10mm
        
        ctx.save();
        ctx.translate(pos.x, pos.y);
        ctx.rotate(-(particleSummary.poseTheta + Math.PI / 2));
        ctx.translate(tipOffset, 0);  // Apply tip offset
        
        ctx.beginPath();
        ctx.moveTo(size, 0);
        ctx.lineTo(-size * 0.6, size * 0.5);
        ctx.lineTo(-size * 0.6, -size * 0.5);
        ctx.closePath();
        
        // Yellow with 0.7 alpha, matching legacy (1, 1, 0, 0.7)
        ctx.fillStyle = "rgba(255, 255, 0, 0.7)";
        ctx.fill();
        
        ctx.restore();
    }

    function drawLandmarks(ctx) {
        if (!landmarkModel)
            return;
        var count = landmarkModel.count || 0;
        var scale = pixelsPerMm();
        for (var i = 0; i < count; ++i) {
            var entry = landmarkModel.get(i);
            if (!entry)
                continue;
            var pos = mapPoint(entry.x, entry.y);
            ctx.save();
            ctx.translate(pos.x, pos.y);
            ctx.rotate(-(entry.theta + Math.PI / 2));
            if (entry.kind === "wall") {
                var halfLength = (entry.length_mm || 100) * 0.5 * scale;
                var halfWidth = Math.max(10 * scale, (entry.width_mm || 50) * 0.5 * scale);
                ctx.strokeStyle = entry.seen ? "rgba(255, 140, 90, 0.9)" : "rgba(120, 80, 60, 0.5)";
                ctx.lineWidth = 2;
                ctx.strokeRect(-halfLength, -halfWidth / 4, halfLength * 2, halfWidth / 2);
            } else {
                var size = 20 * scale;
                ctx.strokeStyle = entry.seen ? "rgba(130, 255, 120, 0.9)" : "rgba(90, 150, 90, 0.6)";
                ctx.lineWidth = 2;
                ctx.strokeRect(-size / 2, -size / 2, size, size);
                if (entry.kind === "aruco" && entry.marker_id !== null && entry.marker_id !== undefined) {
                    ctx.save();
                    ctx.rotate(entry.theta + Math.PI / 2);
                    ctx.fillStyle = entry.seen ? "rgba(255, 255, 255, 0.95)" : "rgba(200, 200, 200, 0.7)";
                    ctx.font = Math.max(10, 14 * scale) + "px sans-serif";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(String(entry.marker_id), 0, size * 0.08);
                    ctx.restore();
                }
            }
            ctx.restore();
        }
    }

    onPaint: {
        var ctx = getContext("2d");
        ctx.reset();
        ctx.fillStyle = backgroundColor;
        ctx.fillRect(0, 0, width, height);

        drawGrid(ctx);
        drawLandmarks(ctx);
        drawParticles(ctx);    // Draw particles first (red, small)
        drawSummary(ctx);      // Draw error ellipse (cyan)
        drawRobot(ctx);        // Draw robot on top (yellow, large)
    }
}
