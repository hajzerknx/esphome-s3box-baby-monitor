# SPDX-License-Identifier: Apache-2.0
import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import display, speaker
from esphome.const import CONF_ID, CONF_NAME
from esphome.core import CORE

CONF_CAMERA_1 = "camera_1"
CONF_CAMERA_2 = "camera_2"
CONF_DISPLAY = "display"
CONF_SPEAKER = "speaker"
CONF_AUDIO_VOLUME = "audio_volume"
CONF_HOST = "host"
CONF_PORT = "port"
CONF_USERNAME = "username"
CONF_PASSWORD = "password"
CONF_PATH = "path"

AUTO_LOAD = ["display", "speaker"]
DEPENDENCIES = ["display", "speaker"]

baby_monitor_ns = cg.esphome_ns.namespace("baby_monitor")
BabyMonitor = baby_monitor_ns.class_("BabyMonitor", cg.Component)

CAMERA_SCHEMA = cv.Schema({
    cv.Required(CONF_NAME): cv.string_strict,
    cv.Required(CONF_HOST): cv.string_strict,
    cv.Optional(CONF_PORT, default=554): cv.port,
    cv.Required(CONF_USERNAME): cv.string,
    cv.Required(CONF_PASSWORD): cv.string,
    cv.Required(CONF_PATH): cv.string_strict,
})

CONFIG_SCHEMA = cv.Schema({
    cv.GenerateID(): cv.declare_id(BabyMonitor),
    cv.Required(CONF_DISPLAY): cv.use_id(display.Display),
    cv.Required(CONF_SPEAKER): cv.use_id(speaker.Speaker),
    cv.Optional(CONF_AUDIO_VOLUME, default="5%"): cv.percentage,
    cv.Required(CONF_CAMERA_1): CAMERA_SCHEMA,
    cv.Required(CONF_CAMERA_2): CAMERA_SCHEMA,
}).extend(cv.COMPONENT_SCHEMA)


def _apply_camera(var, index, config):
    cg.add(var.set_camera_name(index, config[CONF_NAME]))
    cg.add(var.set_camera_host(index, config[CONF_HOST]))
    cg.add(var.set_camera_port(index, config[CONF_PORT]))
    cg.add(var.set_camera_username(index, config[CONF_USERNAME]))
    cg.add(var.set_camera_password(index, config[CONF_PASSWORD]))
    cg.add(var.set_camera_path(index, config[CONF_PATH]))


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    disp = await cg.get_variable(config[CONF_DISPLAY])
    cg.add(var.set_display(disp))

    spk = await cg.get_variable(config[CONF_SPEAKER])
    cg.add(var.set_speaker(spk))
    cg.add(var.set_audio_volume(config[CONF_AUDIO_VOLUME]))

    _apply_camera(var, 0, config[CONF_CAMERA_1])
    _apply_camera(var, 1, config[CONF_CAMERA_2])

    # JPEGDEC decodes the reconstructed RFC 2435 JPEG directly from VGA to QVGA.
    cg.add_library("JPEGDEC", "1.8.4", "https://github.com/bitbank2/JPEGDEC#1.8.4")

    if CORE.is_esp32:
        from esphome.components.esp32 import add_idf_component
        # Official Espressif decoder. The component uses the AAC API directly without an ADF pipeline.
        add_idf_component(name="espressif/esp_audio_codec", ref="2.6.2")
