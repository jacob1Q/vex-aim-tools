import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: root

    property string source: ""
    property real squareSizeMm: 5.0
    property var viewState: null
    property real originX: 0.0
    property real originY: 0.0
    property color backgroundColor: "#0f161f"
    property color borderColor: "#203041"

    implicitWidth: 280
    implicitHeight: 280

    Rectangle {
        anchors.fill: parent
        color: backgroundColor
        border.color: borderColor
        border.width: 1
    }

    Item {
        id: viewport
        anchors.fill: parent
        anchors.margins: 8
        clip: true

        Image {
            id: wavefrontImage
            source: root.source
            smooth: false
            visible: root.source.length > 0
            fillMode: Image.Stretch
            width: root.imageWidth()
            height: root.imageHeight()
            x: root.imageX()
            y: root.imageY()
        }
    }

    Label {
        anchors.centerIn: parent
        visible: root.source.length === 0
        text: qsTr("Wavefront\nunavailable")
        horizontalAlignment: Text.AlignHCenter
        color: "#54657a"
    }

    function pixelsPerMm() {
        if (!root.viewState || root.viewState.zoom === undefined || root.viewState.zoom === null)
            return 0.64;
        return Math.max(0.01, root.viewState.zoom);
    }

    function centerX() {
        if (!root.viewState || root.viewState.centerX === undefined)
            return 0;
        return root.viewState.centerX;
    }

    function centerY() {
        if (!root.viewState || root.viewState.centerY === undefined)
            return 0;
        return root.viewState.centerY;
    }

    function imageScale() {
        return pixelsPerMm() * Math.max(1.0, root.squareSizeMm);
    }

    function imageWidth() {
        if (!wavefrontImage.visible || wavefrontImage.sourceSize.width <= 0)
            return 0;
        return wavefrontImage.sourceSize.width * imageScale();
    }

    function imageHeight() {
        if (!wavefrontImage.visible || wavefrontImage.sourceSize.height <= 0)
            return 0;
        return wavefrontImage.sourceSize.height * imageScale();
    }

    function imageX() {
        return (root.originX - centerX()) * pixelsPerMm() + viewport.width / 2;
    }

    function imageY() {
        return (centerY() - root.originY) * pixelsPerMm() + viewport.height / 2;
    }
}
