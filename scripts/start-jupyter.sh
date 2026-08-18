#!/usr/bin/env bash
nohup "$UV_PROJECT_ENVIRONMENT/bin/jupyter" lab --port 1984 --IdentityProvider.token devcontainer-token --no-browser \
    --JupyterMCPServerExtensionApp.document_id="" &