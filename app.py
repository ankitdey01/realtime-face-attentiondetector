from flask import Flask, render_template, send_from_directory


app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/public/<path:filename>")
def public_file(filename):
    return send_from_directory(app.root_path + "/public", filename)


@app.route("/models/<path:filename>")
def model_file(filename):
    return send_from_directory(app.root_path + "/models", filename)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
