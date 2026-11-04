import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


function viewUrlOf(voxFile) {
  const query = new URLSearchParams(voxFile);
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


function addDownloadButton(node) {
  const button = node.addWidget("button", "Download .vox", null, () => download(node.voxFile));
  button.serialize = false;
  return button;
}


function showDownload(node, voxFile) {
  node.voxFile = voxFile;
  node.downloadButton ??= addDownloadButton(node);
  node.downloadButton.label = `Download ${voxFile.filename}`;
  node.setDirtyCanvas(true, true);
}


app.registerExtension({
  name: "vexel.SaveVoxelsAsVox.download",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SaveVoxelsAsVox") {
      return;
    }

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
