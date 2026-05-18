/**
 * Registra en el proyecto QuPath las clases Gleason (pred + GT) para que aparezcan
 * en el panel "Class list" con color, y puedas usar el ojo para ocultar bloques.
 *
 * Paleta (pred = vivo; GT = más contraste sobre eosina rosada: fríos + más croma):
 *   Clase      Hex pred   RGB pred        Hex GT      RGB GT
 *   NC         #6B6B6B    107,107,107     #5F7382     95,115,130
 *   GG3        #006BA4     0,107,164      #38B2C9     56,178,201
 *   GG4        #E69F00   230,159,  0      #E09128    224,145,40
 *   GG5        #CC3311   204, 51, 17      #AB60C4    171,96,196
 *
 * Requisito: tener un proyecto abierto (File → Open project… o crear uno).
 * Ejecuta este script una vez por proyecto (o cada vez que abras un proyecto nuevo).
 */

import qupath.lib.objects.classes.PathClass

def rgb(int r, int g, int b) {
    return ((0xFF << 24) | (r << 16) | (g << 8) | b) as Integer
}

def project = getProject()
if (project == null) {
    println 'Abre o crea un proyecto QuPath y vuelve a ejecutar este script.'
    println 'Sin proyecto, las anotaciones pueden tener nombre de clase pero no aparecen en la lista de clases del proyecto.'
    return
}

def merged = new LinkedHashSet<PathClass>()
merged.addAll(project.getPathClasses())

// Predicción — colores vivos, distinguibles en H&E
merged.add(PathClass.fromString('NC', rgb(107, 107, 107)))
merged.add(PathClass.fromString('GG3', rgb(0, 107, 164)))
merged.add(PathClass.fromString('GG4', rgb(230, 159, 0)))
merged.add(PathClass.fromString('GG5', rgb(204, 51, 17)))

// GT — gris azulado, teal, ámbar fuerte, violeta (se leen sobre fondo rosado H&E)
merged.add(PathClass.fromString('GT NC', rgb(95, 115, 130)))
merged.add(PathClass.fromString('GT GG3', rgb(56, 178, 201)))
merged.add(PathClass.fromString('GT GG4', rgb(224, 145, 40)))
merged.add(PathClass.fromString('GT GG5', rgb(171, 96, 196)))

def ok = project.setPathClasses(new ArrayList<PathClass>(merged))
println "Clases registradas en el proyecto (setPathClasses=${ok}). Total: ${merged.size()}"
try {
    project.syncChanges()
    println 'Proyecto guardado (syncChanges).'
} catch (Exception e) {
    println "No se pudo guardar el proyecto automáticamente: ${e.message}"
}
