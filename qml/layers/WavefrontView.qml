import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: root

    property string source: ""
    property real squareSizeMm: 5.0
    property var viewState: null
    property color backgroundColor: "#0f161f"
    property color borderColor: "#203041"

    implicitWidth: 280
    implicitHeight: 280

    Rectangle {
        anchors.fill: parent
        color: backgroundColor
        border.color: borderColor
        border.width: 1

        Image {
            id: wavefrontImage
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: parent.verticalCenter
            source: root.source
            fillMode: Image.PreserveAspectFit
            smooth: false
            visible: root.source.length > 0
            width: contentWidth()
            height: contentHeight()

            function contentWidth() {
                if (!visible)
                    return 0;
                var pxPerMm = root.viewState && root.viewState.zoom !== undefined ? root.viewState.zoom : 0.64;
                var cellMm = Math.max(1.0, root.squareSizeMm);
                var imageWidth = sourceSize.width > 0 ? sourceSize.width : implicitWidth;
                var widthPx = imageWidth * cellMm * pxPerMm;
                return Math.min(parent.width - 16, widthPx);
            }

            function contentHeight() {
                if (!visible)
                    return 0;
                var pxPerMm = root.viewState && root.viewState.zoom !== undefined ? root.viewState.zoom : 0.64;
                var cellMm = Math.max(1.0, root.squareSizeMm);
                var imageHeight = sourceSize.height > 0 ? sourceSize.height : implicitHeight;
                var heightPx = imageHeight * cellMm * pxPerMm;
                return Math.min(parent.height - 16, heightPx);
            }
        }

        Label {
            anchors.centerIn: parent
            visible: root.source.length === 0
            text: qsTr("Wavefront\nunavailable")
            horizontalAlignment: Text.AlignHCenter
            color: "#54657a"
        }
    }
}
