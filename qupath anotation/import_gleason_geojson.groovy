/**
 * Importa GeoJSON multiclase (Gleason SICAP) generados por:
 *   scripts/run_pandas_sicap_tile_inference.py  (sin --binary-cancer-mode)
 *
 * Por caso:
 *   case_pred_gleason_annotations.geojson          — predicción
 *   case_gt_gleason_annotations_soft.geojson       — GT suavizado (+ fillOpacity en propiedades)
 *
 * Usa Gson (incluido en QuPath), no groovy.json.JsonSlurper.
 *
 * Ajusta candidateInferenceRoots y ejecuta con una imagen abierta.
 */

import com.google.gson.JsonParser
import qupath.lib.objects.classes.PathClass
import qupath.lib.io.PathIO

/**
 * Añade NC / GG3–GG5 y GT NC / GT GG3–GT GG5 al proyecto para que salgan en "Class list"
 * y puedas ocultar pred vs GT con el ojo.
 */
def registerGleasonPathClassesInProject() {
    def rgb = { int r, int g, int b -> ((0xFF << 24) | (r << 16) | (g << 8) | b) as Integer }
    def project = getProject()
    if (project == null) {
        println 'Aviso: no hay proyecto abierto; las clases no se registrarán en la lista del proyecto.'
        println 'Ejecuta primero qupath anotation/setup_gleason_qupath_classes.groovy con un proyecto abierto.'
        return
    }
    def merged = new LinkedHashSet<PathClass>()
    merged.addAll(project.getPathClasses())
    merged.add(PathClass.fromString('NC', rgb(107, 107, 107)))
    merged.add(PathClass.fromString('GG3', rgb(0, 107, 164)))
    merged.add(PathClass.fromString('GG4', rgb(230, 159, 0)))
    merged.add(PathClass.fromString('GG5', rgb(204, 51, 17)))
    merged.add(PathClass.fromString('GT NC', rgb(95, 115, 130)))
    merged.add(PathClass.fromString('GT GG3', rgb(56, 178, 201)))
    merged.add(PathClass.fromString('GT GG4', rgb(224, 145, 40)))
    merged.add(PathClass.fromString('GT GG5', rgb(171, 96, 196)))
    project.setPathClasses(new ArrayList<PathClass>(merged))
    try {
        project.syncChanges()
    } catch (Throwable ignored) {
        // sin guardar no es crítico para importar
    }
}

// --- CONFIG ---
def repoRoot = 'C:/Users/Aleix/OneDrive - Universitat Politècnica de Catalunya/Escritorio/UNI/TFG/Recerca primers datasets/SicapV2/SICAPv2'
def candidateInferenceRoots = [
    buildFilePath(repoRoot, 'outputs', 'sicap_inference_10x_final_all_folds'),
    buildFilePath(repoRoot, 'outputs', 'pandas_tile_inference_gleason_geojson_10x_final_all_folds'),
    buildFilePath(repoRoot, 'outputs', 'pandas_tile_inference_gleason_geojson')
]
def predGeojsonName = 'case_pred_gleason_annotations.geojson'
def gtGeojsonName = 'case_gt_gleason_annotations_soft.geojson'
def importPrediction = true
def importGroundTruth = false
/** Prefijo para clases GT en QuPath (ej. "GT GG3") */
def gtClassPrefix = 'GT '
def clearExistingAnnotations = true
// -------------

/** Lee FeatureCollection y devuelve el array "features" (puede ser null). */
def readGeoJsonFeatures(File geojsonFile) {
    def text = geojsonFile.getText('UTF-8')
    def root
    try {
        root = JsonParser.parseString(text).getAsJsonObject()
    } catch (Throwable ignored) {
        root = new JsonParser().parse(text).getAsJsonObject()
    }
    if (!root.has('features') || root.get('features').isJsonNull())
        return null
    return root.getAsJsonArray('features')
}

def readClassNameFromFeature(def feature) {
    if (feature == null || !feature.has('properties') || feature.get('properties').isJsonNull())
        return 'Unknown'
    def props = feature.getAsJsonObject('properties')
    if (!props.has('class_name') || props.get('class_name').isJsonNull())
        return 'Unknown'
    return props.get('class_name').getAsString()
}

def assignClassesFromGeoJson(File geojsonFile, String classPrefix) {
    if (!geojsonFile.exists()) {
        println "Missing: ${geojsonFile.getAbsolutePath()}"
        return []
    }
    def feats = readGeoJsonFeatures(geojsonFile)
    def objs = PathIO.readObjects(geojsonFile)
    if (objs == null || objs.isEmpty()) {
        println "No objects from ${geojsonFile.getName()}"
        return []
    }
    def out = []
    for (int i = 0; i < objs.size(); i++) {
        def obj = objs[i]
        def clsName = 'Unknown'
        if (feats != null && i < feats.size()) {
            def feat = feats.get(i).getAsJsonObject()
            clsName = readClassNameFromFeature(feat)
        }
        def fullName = classPrefix + clsName
        def pc = getPathClass(fullName) ?: PathClass.fromString(fullName)
        obj.setPathClass(pc)
        out.add(obj)
    }
    return out
}

def findCaseBase = { String caseId, String requiredFileName ->
    for (root in candidateInferenceRoots) {
        def base = buildFilePath(root, caseId)
        def geojsonFile = new File(buildFilePath(base, requiredFileName))
        if (geojsonFile.exists()) {
            println "Using inference output: ${root}"
            return base
        }
    }
    return null
}

def entry = getProjectEntry()
if (entry == null) {
    print 'No active project entry/image.'
    return
}

def imageName = entry.getImageName()
def dot = imageName.lastIndexOf('.')
def caseId = dot > 0 ? imageName.substring(0, dot) : imageName

println "Image: ${imageName}"
println "Case ID: ${caseId}"

registerGleasonPathClassesInProject()

if (clearExistingAnnotations) {
    def current = getAnnotationObjects()
    if (current != null && !current.isEmpty()) {
        removeObjects(current, true)
    }
}

def base = findCaseBase(caseId, predGeojsonName)
if (base == null) {
    println "No prediction GeoJSON found for case '${caseId}'. Checked:"
    candidateInferenceRoots.each { root ->
        println "  " + buildFilePath(root, caseId, predGeojsonName)
    }
    println "Note: scripts/sicap_infer.py currently writes PNG/NPY mosaics but not GeoJSON annotations."
    return
}
def allAdded = []

if (importPrediction) {
    def predFile = new File(buildFilePath(base, predGeojsonName))
    def predObjs = assignClassesFromGeoJson(predFile, '')
    addObjects(predObjs)
    allAdded.addAll(predObjs)
    println "Prediction: ${predObjs.size()} object(s) from ${predGeojsonName}"
}

if (importGroundTruth) {
    def gtFile = new File(buildFilePath(base, gtGeojsonName))
    def gtObjs = assignClassesFromGeoJson(gtFile, gtClassPrefix)
    addObjects(gtObjs)
    allAdded.addAll(gtObjs)
    println "Ground truth (soft): ${gtObjs.size()} object(s) from ${gtGeojsonName}"
}

println "Total added: ${allAdded.size()}"
