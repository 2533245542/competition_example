//import tech.tablesaw.api.Table
//import tech.tablesaw.io.ReadOptions
//import tech.tablesaw.io.csv.CsvReadOptions
//import java.util.*
//import java.util.stream.IntStream
//
//var readOptions: ReadOptions = CsvReadOptions.builder("../zwpweka/experiments/creditg/out/earlyStoppedAlgorithm.csv").header(false).build()
//val table: Table? = null;
//try {
//    val table = Table.read().usingOptions(readOptions);
//} catch (e: Exception) {
//    e.printStackTrace();
//}
//
//// set names
//val columnNames = Arrays.asList("algName", "optRd", "i", "status");
//val finalTable: Table? = table;
//IntStream.range(0, columnNames.size).forEach { i: Int -> finalTable?.column(i)?.setName(columnNames[i]) }
//
var isMeta = false
var algType: String = if (isMeta) "Meta" else "Base"
print(algType)


