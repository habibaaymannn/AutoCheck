import textwrap
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.ConfigManager import ConfigManager, ConfigParseError, ConfigValidationError


def write_yaml(tmp_path, content: str):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


def test_valid_ml_config(tmp_path):
    path = write_yaml(tmp_path, """
    system:
      execution_mode: ml
      fram_schd: pytorch
      run_id: run1

    ml_model:
      name: main_model

    checkpoint:
      method: time
      interval: 300
      max_session_time: 900
      safety_buffer_seconds: 15
      keep_last: 3
      save_dir: ./checkpoints

    notify:
      email: user@test.com
      on_failure: true
      on_checkpoint: false
    """)

    cm = ConfigManager()
    cm.parse(path)
    assert cm.validate() is True
    assert cm.mode == "ml"


def test_valid_hpc_config(tmp_path):
    path = write_yaml(tmp_path, """
    system:
      execution_mode: hpc
      fram_schd: slurm
      run_id: run2

    hpc:
      tracked_states:
        - name: nodes
          type: int
          source: "sinfo"

    checkpoint:
      method: time
      interval: 60
      max_session_time: 600
      safety_buffer_seconds: 10
      keep_last: 2
      save_dir: ./checkpoints
    """)

    cm = ConfigManager()
    cm.parse(path)
    assert cm.validate() is True
    assert cm.mode == "hpc"


def test_missing_required_section_ml(tmp_path):
    path = write_yaml(tmp_path, """
    system:
      execution_mode: ml
      fram_schd: pytorch
      run_id: run1

    checkpoint:
      method: time
      interval: 300
      max_session_time: 900
      save_dir: ./checkpoints
    """)

    cm = ConfigManager()
    with pytest.raises(ConfigParseError):
        cm.parse(path)


def test_forbidden_section_in_ml_mode(tmp_path):
    path = write_yaml(tmp_path, """
    system:
      execution_mode: ml
      fram_schd: pytorch
      run_id: run1

    ml_model:
      name: main_model

    hpc:
      tracked_states:
        - name: nodes
          type: int
          source: "sinfo"

    checkpoint:
      method: time
      interval: 300
      max_session_time: 900
      save_dir: ./checkpoints
    """)

    cm = ConfigManager()
    with pytest.raises(ConfigParseError):
        cm.parse(path)


def test_unknown_top_level_section(tmp_path):
    path = write_yaml(tmp_path, """
    system:
      execution_mode: ml
      fram_schd: pytorch
      run_id: run1

    ml_model:
      name: main_model

    checkpoint:
      method: time
      interval: 300
      max_session_time: 900
      save_dir: ./checkpoints

    extra_section:
      x: 1
    """)

    cm = ConfigManager()
    with pytest.raises(ConfigParseError):
        cm.parse(path)


def test_invalid_execution_mode(tmp_path):
    path = write_yaml(tmp_path, """
    system:
      execution_mode: abc
      fram_schd: pytorch
      run_id: run1

    ml_model:
      name: main_model

    checkpoint:
      method: time
      interval: 300
      max_session_time: 900
      save_dir: ./checkpoints
    """)

    cm = ConfigManager()
    cm.parse(path)
    with pytest.raises(ConfigValidationError):
        cm.validate()


def test_invalid_checkpoint_method(tmp_path):
    path = write_yaml(tmp_path, """
    system:
      execution_mode: ml
      fram_schd: pytorch
      run_id: run1

    ml_model:
      name: main_model

    checkpoint:
      method: wrong_method
      interval: 300
      max_session_time: 900
      save_dir: ./checkpoints
    """)

    cm = ConfigManager()
    cm.parse(path)
    with pytest.raises(ConfigValidationError):
        cm.validate()


def test_cross_field_checkpoint_relation(tmp_path):
    path = write_yaml(tmp_path, """
    system:
      execution_mode: ml
      fram_schd: pytorch
      run_id: run1

    ml_model:
      name: main_model

    checkpoint:
      method: time
      interval: 1000
      max_session_time: 900
      safety_buffer_seconds: 100
      keep_last: 3
      save_dir: ./checkpoints
    """)

    cm = ConfigManager()
    cm.parse(path)
    with pytest.raises(ConfigValidationError):
        cm.validate()


def test_invalid_hpc_state_type(tmp_path):
    path = write_yaml(tmp_path, """
    system:
      execution_mode: hpc
      fram_schd: slurm
      run_id: run2

    hpc:
      tracked_states:
        - name: nodes
          type: wrong
          source: "sinfo"

    checkpoint:
      method: time
      interval: 60
      max_session_time: 600
      save_dir: ./checkpoints
    """)

    cm = ConfigManager()
    cm.parse(path)
    with pytest.raises(ConfigValidationError):
        cm.validate()
