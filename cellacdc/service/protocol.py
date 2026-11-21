"""Version of the HTTP protocol between Cell-ACDC and its model workers.

The client (the Cell-ACDC GUI or CLI) and the worker
(`cellacdc.service.worker`, running in the model's environment) must use the
same `PROTOCOL_VERSION`. Increase it whenever a change to the endpoints or to
the request/response format makes older clients or workers incompatible.

How the versions are checked:
- The client reads the worker's version with `GET /version` before using it
  (`client.check_worker_version`). Workers with protocol 1 have no such
  endpoint (HTTP 404). A local worker with another version is repaired by
  reinstalling Cell-ACDC in its environment
  (`environment.repair_cellacdc_runtime`) and restarted.
- The client sends its version in the `PROTOCOL_HEADER` header of every
  request, and the worker rejects requests with another version (HTTP 426),
  e.g. from an older Cell-ACDC that does not check the worker's version.
- Before starting a worker, Cell-ACDC checks that the model environment has
  the same Cell-ACDC build as the client, and reinstalls it otherwise
  (`environment.ensure_cellacdc_cli`).

History:
    1: arguments as JSON text in form fields.
    2: every argument in a binary attachment (see `payload.py`), files on the
       client's disk are uploaded automatically, `GET /version`.
"""

PROTOCOL_VERSION = 2

PROTOCOL_HEADER = 'X-Cell-ACDC-Protocol'


def headers():
    """HTTP headers that the client sends with every request."""
    return {PROTOCOL_HEADER: str(PROTOCOL_VERSION)}
