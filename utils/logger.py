from constants import PATH_LOGGER
import logging

# Crear un logger
logger = logging.getLogger("Feature_density_UQ")
logger.setLevel(logging.INFO)

# Crear un handler para escribir en un archivo
file_handler = logging.FileHandler(PATH_LOGGER)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Crear otro handler para la salida estándar (consola)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Añadir ambos handlers al logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)