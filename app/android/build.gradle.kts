allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// Force every Android module (incl. the onnxruntime plugin, shipped at compileSdk 33)
// to compile against API 36, so transitive androidx deps that require 34+ resolve.
// Must run AFTER the module's own build.gradle sets its compileSdk, hence afterEvaluate
// (guarded against modules that are already evaluated due to evaluationDependsOn).
subprojects {
    val forceSdk = {
        (extensions.findByName("android") as? com.android.build.gradle.BaseExtension)
            ?.compileSdkVersion(36)
        Unit
    }
    if (state.executed) forceSdk() else afterEvaluate { forceSdk() }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
