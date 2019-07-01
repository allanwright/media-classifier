FROM microsoft/cntk:2.6-cpu-python3.5

COPY .devcontainer/environment.yml /workspace/
COPY .devcontainer/environment.sh /workspace/

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils 2>&1 \
    #
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    && apt-get -y install git procps lsb-release \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && /workspace/environment.sh

ENV SHELL /bin/bash
