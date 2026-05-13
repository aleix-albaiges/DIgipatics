/**
 * Exporta anotaciones de un proyecto QuPath al mismo layout que la inferencia binaria:
 *   <outputRoot>/<case_id>/case_pred_binary_annotations.geojson
 *
 * Así puedes reutilizar el mismo script de importación.
 */

import java.io.File
import static qupath.lib.gui.scripting.QPEx.*

// --- CONFIG ---
def targetClasses = ["Cancer"]
def outputRoot = buildFilePath(PROJECT_BASE_DIR, "geojson_binary_like_inference")
def outputGeojsonName = "case_pred_binary_annotations.geojson"
// -------------

def project = getProject()
if (project == null) {
    print "No project open."
    return
}

def rootDir = new File(outputRoot)
if (!rootDir.exists()) {
    rootDir.mkdirs()
}

for (entry in project.getImageList()) {
    def imageName = entry.getImageName()
    def dot = imageName.lastIndexOf('.')
    def caseId = dot > 0 ? imageName.substring(0, dot) : imageName
    println "Processing ${imageName} -> case_id=${caseId}"

    def hierarchy = entry.readHierarchy()
    def annotations = hierarchy.getAnnotationObjects()
    def selected = annotations.findAll {
        it.getPathClass() != null && targetClasses.contains(it.getPathClass().getName())
    }

    if (selected.isEmpty()) {
        println "  No matching annotations."
        continue
    }

    def caseDir = new File(rootDir, caseId)
    if (!caseDir.exists()) {
        caseDir.mkdirs()
    }
    def outFile = new File(caseDir, outputGeojsonName)

    try {
        qupath.lib.gui.scripting.QPEx.exportObjectsToGeoJson(selected, outFile.getAbsolutePath(), "FEATURE_COLLECTION")
        println "  Exported ${selected.size()} annotation(s) to ${outFile.getAbsolutePath()}"
    } catch (Exception e) {
        println "  ERROR exporting ${imageName}: ${e.getMessage()}"
    }
}

println "Export complete."
