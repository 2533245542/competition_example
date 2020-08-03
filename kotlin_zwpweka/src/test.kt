import java.io.File
import java.io.FileWriter
import java.io.IOException

fun main() {
    print("hello world")
}

fun clearFile(fileName: String?) {
    val file = File(fileName)
    try {
        val fw = FileWriter(file, false)
        fw.close()
    } catch (e: IOException) {
        e.printStackTrace()
    }
}


