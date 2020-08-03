import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class test {
    public static void main(String[] args) {
        Logger logger = LogManager.getLogger("technique32");
//        Logger logger = LogManager.getLogger();
        logger.fatal("fatal");
        logger.error("error");
        logger.warn("warn");
        logger.info("info");
        logger.debug("debug");
        logger.trace("trace");
    }
}

