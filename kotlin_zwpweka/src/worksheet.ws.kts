import java.nio.file.Paths

val path = Paths.get("").toAbsolutePath().toString()
println("Working Directory = $path")
