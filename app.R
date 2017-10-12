library(shiny)
aa <- getwd()
ui <- navbarPage(
  #parameters
  header = aa,
  footer = "footer",
  collapsible = TRUE,
  
  
  "Microbiome Analysis",
  tabPanel(
    "Upload and Filter Data",
    sidebarLayout(
      sidebarPanel(
        fileInput(
          "file",
          "Choose File",
          accept = c("text/csv",
                     "text/comma-separated-values,text/plain",
                     ".csv")
        ),
        radioButtons("plotType",
                     "Plot type",
                     c("Scatter" = "p", "Line" = "l"))
      ),
      mainPanel(plotOutput("plot"))
    )
  ),
  tabPanel("miLineage",
           sidebarLayout(
             sidebarPanel(radioButtons(
               "plotType",
               "Plot type",
               c("Scatter" = "p", "Line" = "l")
             )),
             mainPanel(plotOutput("miPlot"))
           )),
  navbarMenu("Help",
             tabPanel("Manual"),
             tabPanel("Contact"))
)

server <- function(input, output) {
  file <- reactive({
    req(input$file)
  })
  
  output$miPlot <- renderPlot({
    # pp <- readPNG("a.png")
    # grid::grid.raster(pp)
    # grid::grid.raster(pp, width = 1.5, height = 1.5)

    source("scripts/script-miLineage.R", local = TRUE)
    grid::grid.raster(miPlot)
  })
}

shinyApp(ui = ui, server = server)
