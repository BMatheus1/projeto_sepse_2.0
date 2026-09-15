"""Import an existing adapter from a Colab evidence ZIP without overwriting evidence."""
from __future__ import annotations

import io
import json
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def import_adapter_zip(payload: bytes, destination: Path) -> Path:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Destino existente preservado: {destination}. Use outro caminho para importar.")
    with ZipFile(io.BytesIO(payload)) as archive:
        entries = {}
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                    or any(":" in part for part in path.parts)
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError("ZIP contém caminho inseguro ou link simbólico.")
            if not info.is_dir():
                if path in entries:
                    raise ValueError("ZIP contém caminhos duplicados.")
                entries[path] = info
        candidates = [p.parent for p in entries if p.name == "adapter_config.json"
                      and any(p.parent / w in entries for w in
                              ("adapter_model.safetensors", "adapter_model.bin"))]
        if len(candidates) != 1:
            raise ValueError("Envie ZIP com exatamente um adapter (configuração e pesos juntos).")
        root = candidates[0]
        config = json.loads(archive.read(entries[root / "adapter_config.json"]))
        if not isinstance(config, dict) or not config.get("base_model_name_or_path") or not config.get("peft_type"):
            raise ValueError("adapter_config.json sem identificação do modelo base/PEFT.")
        selected = {p: info for p, info in entries.items() if p.parent == root}
        if not any(info.file_size > 0 for p, info in selected.items()
                   if p.name in ("adapter_model.safetensors", "adapter_model.bin")):
            raise ValueError("Arquivo de pesos vazio.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staging = Path(temporary) / "adapter"
            staging.mkdir()
            for path, info in selected.items():
                with archive.open(info) as source, (staging / path.name).open("xb") as target:
                    shutil.copyfileobj(source, target)
            if destination.exists():
                raise FileExistsError(f"Destino existente preservado: {destination}")
            staging.rename(destination)
    return destination
