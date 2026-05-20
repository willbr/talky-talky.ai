"""Flask app factory and routes for talky-talky.ai."""
from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template

import inventory

log = logging.getLogger("talky")


def create_app() -> Flask:
    flask_app = Flask(__name__, static_folder="static", template_folder="templates")
    logging.basicConfig(level=logging.INFO)

    @flask_app.route("/")
    def index() -> str:
        return render_template("index.html")

    @flask_app.route("/api/inventory")
    def api_inventory():
        return jsonify({"words": inventory.list_words()})

    return flask_app


if __name__ == "__main__":
    create_app().run(debug=True, port=5000)
