import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


const NODE_TYPES = new Set(["SaveVoxelsAsVox", "SaveVoxelsToVoxCollection"]);
const LOADER_TYPE = "LoadNumberedImage";
const HISTORY_LENGTH = 64;


function viewUrlOf(voxFile) {
  const query = new URLSearchParams({ ...voxFile, rand: Math.random() });
  return api.apiURL(`/view?${query}`);
}


function download(voxFile) {
  const link = document.createElement("a");
  link.href = viewUrlOf(voxFile);
  link.download = voxFile.filename;
  document.body.append(link);
  link.click();
  link.remove();
}


function voxFileOfEntry(entry, nodeId) {
  const isSaveNode = NODE_TYPES.has(entry.prompt?.[2]?.[nodeId]?.class_type);
  return isSaveNode ? entry.outputs?.[nodeId]?.vox_files?.[0] : undefined;
}


async function latestVoxFileOf(nodeId) {
  const response = await api.fetchApi(`/history?max_items=${HISTORY_LENGTH}`);
  const entries = Object.values(await response.json()).reverse();
  return entries.map(entry => voxFileOfEntry(entry, nodeId)).find(voxFile => voxFile);
}


async function downloadLatest(node) {
  const voxFile = node.voxFile ?? await latestVoxFileOf(String(node.id));
  if (!voxFile) {
    alert("Nothing saved by this node yet, run the workflow first.");
    return;
  }
  showDownload(node, voxFile);
  download(voxFile);
}


function showDownload(node, voxFile) {
  node.voxFile = voxFile;
  node.downloadButton.label = `Download ${voxFile.filename}`;
  node.setDirtyCanvas(true, true);
}


function addDownloadButton(node) {
  node.downloadButton = node.addWidget("button", "Download .vox", null, () => downloadLatest(node));
  node.downloadButton.serialize = false;
}


function incrementByDefault(node) {
  const control = node.widgets?.find(widget => widget.name === "control_after_generate");
  if (control) {
    control.value = "increment";
  }
}


async function uploadZip(node, file) {
  const body = new FormData();
  body.append("image", file);
  body.append("overwrite", "true");
  const response = await api.fetchApi("/upload/image", { method: "POST", body });
  if (response.status !== 200) {
    alert(`Uploading ${file.name} failed: ${response.status} ${response.statusText}`);
    return;
  }
  const uploaded = await response.json();
  const value = uploaded.subfolder ? `${uploaded.subfolder}/${uploaded.name}` : uploaded.name;
  selectZip(node, value);
}


function selectZip(node, value) {
  const zipWidget = node.widgets.find(widget => widget.name === "zip");
  if (!zipWidget.options.values.includes(value)) {
    zipWidget.options.values.push(value);
  }
  zipWidget.value = value;
  const index = node.widgets.find(widget => widget.name === "index");
  index.value = 0;
  node.setDirtyCanvas(true, true);
}


function chooseZip(node) {
  const chooser = document.createElement("input");
  chooser.type = "file";
  chooser.accept = ".zip,application/zip";
  chooser.addEventListener("change", () => {
    if (chooser.files.length > 0) {
      uploadZip(node, chooser.files[0]);
    }
  });
  chooser.click();
}


function addUploadButton(node) {
  const button = node.addWidget("button", "Upload zip", null, () => chooseZip(node));
  button.serialize = false;
}


app.registerExtension({
  name: "vexel.LoadNumberedImage.increment",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== LOADER_TYPE) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      incrementByDefault(this);
      addUploadButton(this);
    };
  },
});


app.registerExtension({
  name: "vexel.SaveVoxelsAsVox.download",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_TYPES.has(nodeData.name)) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      addDownloadButton(this);
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      const voxFile = output?.vox_files?.[0];
      if (voxFile) {
        showDownload(this, voxFile);
      }
    };
  },
});
