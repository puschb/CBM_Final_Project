# Imformation_Spread_Model
Computational Behavior Modelling Final Project

# NLP Development Environment

This repository includes a Docker container for development work when not using Rivanna.

## Prerequisites

Before you begin, ensure you have the following installed:

- [Docker](https://www.docker.com/get-started)
- [Visual Studio Code](https://code.visualstudio.com/)
- Docker Extension for VS Code
- Dev Containers Extension for VS Code

## Instructions

Follow these steps to set up and run the development environment:

1. **Install Extensions**  
   In Visual Studio Code, make sure to add the following extensions:
   - Docker
   - Dev Containers

2. **Build and Run the Docker Container**  
   Open your terminal and navigate to the project directory. Run the following command:
   ```bash
   docker compose up --build -d

- **`docker compose up`**: Starts the container.
- **`--build`**: Rebuilds the image if it is not already built.
- **`-d`**: Detaches the container from the terminal, allowing you to continue using it.

3. **Attach to the Running Container**
    In VS Code, click the icon in the bottom left that looks like `><`. Select **"Attach to Running Container..."** and choose the `NLP_dev` container. This will open a new VS Code window connected to the container.

4. **Set Up the Environment in the Container**
    Once the new window is open, perform the following actions:
    - Click on **Extensions** and install the Python extension (this only needs to be done the first time you open the container).
    - If you do not see the current directory, navigate to it by clicking **Open Folder**, backing out one step, and selecting `/code`.
