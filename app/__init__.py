#!/usr/bin/env python
# -*- coding: utf-8 -*-
import locale
import os
import logging
# from logging.handlers import SMTPHandler, RotatingFileHandler
from config import Config

from flask import Flask, jsonify


class ORIMonitorError(Exception):
    """API error class.
    :param msg: the message that should be returned to the API user.
    :param status_code: the HTTP status code of the response
    """

    def __init__(self, msg, status_code):
        self.msg = msg
        self.status_code = status_code

    def __str__(self):
        return repr(self.msg)

    @staticmethod
    def serialize_error(e):
        return jsonify(dict(status='error', error=e.msg)), e.status_code


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    app.errorhandler(ORIMonitorError)(ORIMonitorError.serialize_error)

    def add_cors_headers(resp):
        resp.headers['Access-Control-Allow-Origin'] = '*'
        # See https://stackoverflow.com/questions/12630231/how-do-cors-and-access-control-allow-headers-work
        resp.headers['Access-Control-Allow-Headers'] = 'origin, content-type, accept'
        return resp

    app.after_request(add_cors_headers)

    return app

logging.basicConfig(
    format='%(asctime)s.%(msecs)s:%(name)s:%(thread)d:%(levelname)s:%(process)d:%(message)s',
    level=logging.INFO)


app = create_app()

from app import routes
