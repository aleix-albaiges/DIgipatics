/**
 * Importa las anotaciones GeoJSON generadas por:
 * scripts/run_pandas_sicap_tile_inference.py --binary-cancer-mode
 *
 * Estructura esperada:
 *   <inferenceOutputRoot>/<case_id>/case_pred_binary_annotations.geojson
 *
 * Uso:
 * 1) Abre una imagen en QuPath.
 * 2) Ajusta inferenceOutputRoot.
 * 3) Ejecuta este script.
 */

import qupath.lib.objects.classes.PathClass
import qupath.lib.objects.PathObject
import qupath.lib.io.PathIO

// --- CONFIG ---
def inferenceOutputRoot = 'C:/Users/Aleix/OneDrive - Universitat Politècnica de Catalunya/Escritorio/UNI/TFG/Recerca primers datasets/SicapV2/SICAPv2/outputs/pandas_tile_inference_binary_geojson'
def geojsonFileName = 'case_pred_binary_annotations.geojson'
def targetClassName = 'Cancer'
def clearExistingAnnotations = false
// -------------

def entry = getProjectEntry()
if (entry == null) {
    print 'No active project entry/image.'
    return
}

def imageName = entry.getImageName()
def dot = imageName.lastIndexOf('.')
def caseId = dot > 0 ? imageName.substring(0, dot) : imageName
def geojsonPath = buildFilePath(inferenceOutputRoot, caseId, geojsonFileName)
def geojsonFile = new File(geojsonPath)

println "Image: ${imageName}"
println "Case ID: ${caseId}"
println "Expected GeoJSON: ${geojsonFile.getAbsolutePath()}"

if (!geojsonFile.exists()) {
    print "GeoJSON not found for case '${caseId}'."
    return
}

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