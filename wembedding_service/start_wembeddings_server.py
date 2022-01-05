#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Word embeddings server.

Example setup:
$ venv/bin/python ./wembeddings_server.py

Example call:
$ curl --data-binary @examples/request.json localhost:8000/wembeddings | xxd
"""

import signal
import sys
import threading

import numpy as np

import wembeddings.wembeddings as wembeddings
import wembeddings.wembeddings_server as wembeddings_server

if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Port to use")
    parser.add_argument("--dtype", default="float16", type=str, help="Dtype to serve the embeddings as")
    parser.add_argument("--logfile", default=None, type=str, help="Log path")
    parser.add_argument("--preload_models", default=[], nargs="*", type=str, help="Models to preload, or `all`")
    parser.add_argument("--threads", default=4, type=int, help="Threads to use")
    args = parser.parse_args()
    args.dtype = getattr(np, args.dtype)

    # Log stderr to logfile if given
    if args.logfile is not None:
        sys.stderr = open(args.logfile, "a", encoding="utf-8")

    # Create the server and its own thread
    server = wembeddings_server.WEmbeddingsServer(
        args.port,
        args.dtype,
        lambda: wembeddings.WEmbeddings(threads=args.threads, preload_models=args.preload_models),
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("Starting WEmbeddings server on port {}.".format(args.port), file=sys.stderr)
    print("To stop it gracefully, either send SIGINT (Ctrl+C) or SIGUSR1.", file=sys.stderr, flush=True)

    # Wait until the server should be closed
    signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT, signal.SIGUSR1])
    signal.sigwait([signal.SIGINT, signal.SIGUSR1])
    print("Initiating shutdown of the WEmbeddings server.", file=sys.stderr, flush=True)
    server.shutdown()
    print("Stopped handling new requests, processing all current ones.", file=sys.stderr, flush=True)
    server.server_close()
    print("Finished shutdown of the WEmbeddings server.", file=sys.stderr, flush=True)
