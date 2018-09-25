import json
from functools import wraps
import logging

from flask import render_template, request, redirect, url_for, flash, Markup, jsonify

from app import app, ORIMonitorError


@app.route("/")
def index():
    return jsonify({})


if __name__ == "__main__":
    app.run(threaded=True)
