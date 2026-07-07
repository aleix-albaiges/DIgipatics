/**
 * Importa las anotaciones GeoJSON generadas por:
 * scripts/run_pandas_sicap_tile_inference.py --binary-cancer-mode
 *
 * Estructura esperada:
 *   <inferenceOutputRoot>/<case_id>/case_pred_binary_annotations.geojson
 *
 * Uso:
 * 1) Abre una imagen en QuPath.
 * 2) Ajusta candidateInferenceRoots si cambias la carpeta de salida.
 * 3) Ejecuta este script.
 */

import qupath.lib.objects.classes.PathClass
import qupath.lib.objects.PathObject
import qupath.lib.io.PathIO

// --- CONFIG ---
def repoRoot = 'C:/Users/Aleix/OneDrive - Universitat Politècnica de Catalunya/Escritorio/UNI/TFG/Recerca primers datasets/SicapV2/SICAPv2'
def candidateInferenceRoots = [
    buildFilePath(repoRoot, 'outputs', 'pandas_tile_inference_binary_geojson_all'),
    buildFilePath(repoRoot, 'outputs', 'pandas_tile_inference_binary_geojson')
]
def geojsonFileName = 'case_pred_binary_annotations.geojson'
def targetClassName = 'Cancer'
def clearExistingAnnotations = true
// -------------

def entry = getProjectEntry()
if (entry == null) {
    print 'No active project entry/image.'
    return
}

def imageName = entry.getImageName()
def dot = imageName.lastIndexOf('.')
def caseId = dot > 0 ? imageName.substring(0, dot) : imageName
def geojsonFile = null
def selectedRoot = null
for (root in candidateInferenceRoots) {
    def candidate = new File(buildFilePath(root, caseId, geojsonFileName))
    if (candidate.exists()) {
        geojsonFile = candidate
        selectedRoot = root
        break
    }
}

println "Image: ${imageName}"
println "Case ID: ${caseId}"

if (geojsonFile == null) {
    println "GeoJSON not found for case '${caseId}'. Checked:"
    candidateInferenceRoots.each { root ->
        println "  " + buildFilePath(root, caseId, geojsonFileName)
    }
    return
}

println "Using inference output: ${selectedRoot}"
println "GeoJSON: ${geojsonFile.getAbsolutePath()}"

def imported = PathIO.readObjects(geojsonFile)
if (imported == null || imported.isEmpty()) {
    print "GeoJSON found but no objects were imported."
    return
}

def pathClass = getPathClass(targetClassName) ?: PathClass.fromString(targetClassName)
def annotations = imported.findAll { it.getROI() != null }.collect { obj ->
    obj.setPathClass(pathClass)
    return obj
}

if (clearExistingAnnotations) {
    def current = getAnnotationObjects()
    if (current != null && !current.isEmpty()) {
        removeObjects(current, true)
    }
}

addObjects(annotations)
println "Imported ${annotations.size()} annotation(s) from ${geojsonFile.getName()}."
