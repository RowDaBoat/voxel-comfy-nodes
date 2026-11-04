import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


const NODE_TYPES = new Set(["SaveVoxelsAsVox", "SaveVoxelsToVoxCollection"]);
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
