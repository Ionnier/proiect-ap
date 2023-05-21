const path = require('path')
const express = require('express')
const fs = require('fs')
const tmp = require('tmp');
const exec = require('child_process').exec;

var args = process.argv.slice(2);
if (args.length != 2) {
    console.log("Args: pythonCommand pythonSorceDirectory")
    process.exit(1)
}
pythonCommand = args[0]
sourceDirectory = args[1]

const app = express()
app.use(require("cors")())
app.use(express.json());
app.use(express.urlencoded({
    extended: true
}));
app.set("view engine", "ejs")

app.post("/run/:method/", (req, res) => {
    let method = req.params.method
    let command = ""
    switch (method) {
        case "jacobi": {
            command += `${path.join(sourceDirectory, "jacobi", "jpp.py")}`
            break;
        }
        case "gaussian": {
            command += `${path.join(sourceDirectory, "gaussian_elimination", "gaussian_elimination.py")}`
            break;
        }
        case "seidel": {
            command += `${path.join(sourceDirectory, "gauss_seidel", "mpi_gauss_seidel.py")}`
            break;
        }
        default: {
            return res.status(400).end()
        }
    }
    const processors = req.body.processors
    const inputFile = req.body.input

    if (processors == undefined || inputFile == undefined) {
        return res.status(400).end()
    }

    const tmpobj = tmp.fileSync();
    console.log('File: ', tmpobj.name);
    console.log('Filedescriptor: ', tmpobj.fd);
    fs.writeFileSync(tmpobj.fd, inputFile)
    command = `mpirun -n ${processors} ${pythonCommand} "${command.replace(" ", "\ ")}" ${tmpobj.name}`

    switch (method) {
        case "jacobi":
        case "seidel": {
            command += ` ${req.body.iterations ?? 40}`
            break;
        }
    }
    console.log(command)
    exec(command, (error, stdout, stderr) => {
        res.status(200).json({ error, stdout, stderr })
    })
})

app.get("/", (req, res) => {
    res.render("index")
})

app.listen(process.env.PORT || "3000")
