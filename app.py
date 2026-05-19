from flask import Flask, render_template, request
import os
from model_utils import run_inference

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        file = request.files["image"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            pred, score, eds, result_img = run_inference(filepath)

            return render_template(
                "index.html",
                prediction=pred,
                score=round(score, 3),
                eds=round(eds, 3),
                uploaded_image=filepath,
                result_image=result_img
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)