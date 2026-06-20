import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Window {
    width: 1600
    height: 900
    visible: true
    title: "Hello World"

    readonly property list<string> texts: ["Hallo Welt", "Hei maailma",
                                           "Hola Mundo", "Привет мир",
                                           "Hello World", "Ciao Mondo Crudele"]

    function setText() {
        var i = Math.round(Math.random() * 5)
        text.text = texts[i]
    }

    ColumnLayout {
        anchors.fill:  parent

        Text {
            id: text
            text: "Hello World"
            Layout.alignment: Qt.AlignHCenter
        }
        Button {
            text: "Click me"
            Layout.alignment: Qt.AlignHCenter
            onClicked:  setText()
        }
    }
}