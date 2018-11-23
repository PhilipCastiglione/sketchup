import os
import uuid

class WebSpike:
    def __init__(self, detections, prediction_path):
        self.detections = detections
        self.prediction_path = prediction_path

    def generate(self):
        html_detections = '\n'.join([self._map(d) for d in self.detections])
        head = "<head>\n<style>\n.el:hover {\nborder: 1px solid red\n}\n</style>\n</head>"
        html = "<!DOCTYPE html>\n{}<body>\n{}\n</body>\n</html>".format(head, html_detections)
        with open(os.path.join(self.prediction_path, 'prediction.html'), 'w') as f:
            f.write(html)

    def _map(self, detection):
        return {
            "Button": self._button,
            "CheckBox": self._checkbox,
            "ComboBox": self._combobox,
            "Heading": self._heading,
            "Image": self._image,
            "Label": self._label,
            "Link": self._link,
            "Paragraph": self._paragraph,
            "RadioButton": self._radiobutton,
            "TextBox": self._textbox
        }[detection.label]().format(self._style(detection))

    def _button(self):
        return "<button type='button' {}>Button</button>"

    def _checkbox(self):
        return "<div {}><input type='checkbox'/><label>Checkbox</label></div>"

    def _combobox(self):
        return "<select {}><option value='Select From Dropdown'></select>"

    def _heading(self):
        return "<h1 {}>Heading</h1>"

    def _image(self):
        return "<img src='https://via.placeholder.com/100.png' {}/>"

    def _label(self):
        return "<span {}>Label</span>"

    def _link(self):
        return "<a src='#' {}>Link</a>"

    def _paragraph(self):
        return "<p {}>Paragraph</p>"

    def _radiobutton(self):
        return "<div {}><input type='radio'/><label>Radio Button</label></div>"

    def _textbox(self):
        return "<input type='text' {} />"

    def _style(self, detection):
        style = "class='el' "
        style += "style='"
        style += "display:block;"
        style += "position:absolute;"
        style += "top:{}px;".format(detection.y1)
        style += "left:{}px;".format(detection.x1)
        style += "height:{}px;".format(detection.y2 - detection.y1)
        style += "width:{}px;".format(detection.x2 - detection.x1)
        style += "'"
        return style
