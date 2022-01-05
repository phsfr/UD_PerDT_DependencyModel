#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Word embeddings server class."""

import http.server
import json
import socketserver
import sys
import threading
import urllib.parse

import numpy as np

class WEmbeddingsServer(socketserver.ThreadingTCPServer):

    class WEmbeddingsRequestHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def respond(request, content_type, code=200):
            request.close_connection = True
            request.send_response(code)
            request.send_header("Connection", "close")
            request.send_header("Content-Type", content_type)
            request.send_header("Access-Control-Allow-Origin", "*")
            request.end_headers()

        def respond_error(request, message, code=400):
            request.respond("text/plain", code)
            request.wfile.write(message.encode("utf-8"))

        def do_POST(request):
            try:
                request.path = request.path.encode("iso-8859-1").decode("utf-8")
                url = urllib.parse.urlparse(request.path)
            except:
                return request.respond_error("Cannot parse request URL.")

            # Handle /wembeddings
            if url.path == "/wembeddings":
                if request.headers.get("Transfer-Encoding", "identity").lower() != "identity":
                    return request.respond_error("Only 'identity' Transfer-Encoding of payload is supported for now.")

                if "Content-Length" not in request.headers:
                    return request.respond_error("The Content-Length of payload is required.")

                try:
                    length = int(request.headers["Content-Length"])
                    data = json.loads(request.rfile.read(length))
                    model, sentences = data["model"], data["sentences"]
                except:
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()
                    return request.respond_error("Malformed request.")

                try:
                    with request.server._wembeddings_mutex:
                        sentences_embeddings = request.server._wembeddings.compute_embeddings(model, sentences)
                except:
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()
                    return request.respond_error("An error occurred during wembeddings computation.")

                request.respond("application/octet_stream")
                for sentence_embedding in sentences_embeddings:
                    np.lib.format.write_array(request.wfile, sentence_embedding.astype(request.server._dtype), allow_pickle=False)

            # URL not found
            else:
                request.respond_error("No handler for the given URL '{}'".format(url.path), code=404)

    daemon_threads = False

    def __init__(self, port, dtype, wembeddings_lambda):
        self._dtype = dtype

        # Create the WEmbeddings object its mutex
        self._wembeddings = wembeddings_lambda()
        self._wembeddings_mutex = threading.Lock()

        # Initialize the server
        super().__init__(("", port), self.WEmbeddingsRequestHandler)

    def server_bind(self):
        import socket
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        super().server_bind()

    def service_actions(self):
        if isinstance(getattr(self, "_threads", None), list):
            if len(self._threads) >= 1024:
                self._threads = [thread for thread in self._threads if thread.is_alive()]
