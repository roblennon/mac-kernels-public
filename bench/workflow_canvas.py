"""Serialize explicit API graphs as editable Comfy canvases."""


def ui_graph(api, schemas):
    nodes, links = {}, []
    for node_id, entry in api.items():
        sockets, outputs, widget_keys = schemas[entry['class_type']]
        # A widget fed by a link keeps a placeholder value and gains a widget input socket.
        linked = [key for key in widget_keys if not key.startswith('@') and isinstance(entry['inputs'][key], list)]
        widgets = ['fixed' if key == '@seed' else 'image' if key == '@upload' else '' if key in linked
                   else entry['inputs'][key] for key in widget_keys]
        order = int(node_id) - 1
        nodes[node_id] = dict(id=int(node_id), type=entry['class_type'], title=entry['_meta']['title'],
            pos=[(order // 4) * 400, (order % 4) * 420], size=[360, 360], flags={}, order=order, mode=0,
            inputs=[{'name': key, 'type': kind, 'link': None} for key, kind in sockets]
                   + [{'name': key, 'type': None, 'widget': {'name': key}, 'link': None} for key in linked],
            outputs=[{'name': key, 'type': kind, 'links': None} for key, kind in outputs],
            properties={'Node name for S&R': entry['class_type']}, widgets_values=widgets)
    for node_id, entry in api.items():
        for slot, socket in enumerate(nodes[node_id]['inputs']):
            value = entry['inputs'].get(socket['name'])
            if not isinstance(value, list):
                continue
            source, source_slot = value
            link_id = len(links) + 1
            output = nodes[source]['outputs'][source_slot]
            if socket['type'] is None:
                socket['type'] = output['type']  # widget inputs take the linked output's type
            elif socket['type'] != '*' and output['type'] != socket['type']:
                raise ValueError('Incompatible graph sockets')
            links.append([link_id, int(source), source_slot, int(node_id), slot, output['type']])
            socket['link'] = link_id
            output['links'] = (output['links'] or []) + [link_id]
    return dict(last_node_id=len(nodes), last_link_id=len(links), nodes=list(nodes.values()), links=links,
                groups=[], config={}, extra={'ds': {'scale': 1., 'offset': [50, 80]}}, version=.4)

